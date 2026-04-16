"""
traffic_demo_streamlit.py
Streamlit GUI for Traffic Detection + Weather-aware timing + Heatmap + Exports
Merged with live frame display and an online-learning ML pipeline (SGDRegressor + StandardScaler).
"""

import os
import io
import time
import math
import json
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from datetime import datetime
from collections import deque, defaultdict

# ML
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# YOLO + SORT (make sure yolov8n.pt exists and sort package is present)
from ultralytics import YOLO
import sys
sys.path.append("./sort")
from sort import Sort  # __init__.py should export Sort

# ----------------- Config / Defaults -----------------
DEFAULT_VIDEO = "traffic2.mp4"
TARGET_CLASSES = {"car","motorcycle","bus","truck","ambulance"}
DISTANCE_FACTOR = 500
JSON_FILE = "traffic_output.json"
SPEED_LIMIT_KMH = 60
PIXELS_PER_METER = 10
OVERSPEEDING_FILE = "overspeeding_vehicles.json"
HEATMAP_FILE = "traffic_heatmap.json"

WEATHER_POLL_INTERVAL_SEC = 300
WEATHER_GREEN_EXTENSION_FACTOR = {"heavy_rain": 1.6, "light_rain": 1.2, "fog": 1.5, "snow": 1.4, "clear": 1.0}
RUSH_HOUR_MULTIPLIER_PEAK = 1.5
SHORT_TERM_CONGESTION_MULTIPLIER = {"low": 1.0, "moderate": 1.2, "high": 1.4}
SLIDING_WINDOW_SECONDS = 60

MIN_ROWS_TO_TRAIN = 30
ML_MAX_GREEN = 90
ML_MIN_GREEN = 3
ML_MODEL_FILE = "ml_model.joblib"
ML_SCALER_FILE = "ml_scaler.joblib"

# ----------------- Utilities -----------------
def open_video_capture(path_or_index):
    """Robustly open a video path or webcam index; returns (cap, error_message_or_None)."""
    try:
        idx = int(path_or_index)
        use_index = True
    except Exception:
        use_index = False

    if use_index:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap, None
        return None, f"Cannot open webcam index {idx}."

    candidates = [path_or_index]
    if not os.path.isabs(path_or_index):
        candidates.append(os.path.join(os.path.dirname(__file__), path_or_index))
        candidates.append(os.path.join("E:\\SIH_PROJECT", path_or_index))
    for p in candidates:
        if not p:
            continue
        if not os.path.exists(p):
            continue
        cap = cv2.VideoCapture(p)
        if cap.isOpened():
            return cap, None
        # try ffmpeg backend option if available
        try:
            cap = cv2.VideoCapture(p, cv2.CAP_FFMPEG)
            if cap.isOpened():
                return cap, None
        except Exception:
            pass
    tried = "\n".join(f"- {c}" for c in candidates)
    return None, f"Cannot open video file. Tried:\n{tried}\nPossible causes: file not found, corrupted file, missing codec/ffmpeg."

def calculate_speed(prev_center, curr_center, frames_diff, fps, pixels_per_meter):
    if prev_center is None or frames_diff == 0:
        return 0.0
    pixel_distance = math.hypot(curr_center[0]-prev_center[0], curr_center[1]-prev_center[1])
    distance_m = pixel_distance / pixels_per_meter
    time_s = frames_diff / (fps or 30.0)
    if time_s == 0:
        return 0.0
    return (distance_m / time_s) * 3.6

def fetch_weather_condition_cached(api_key, lat, lon, cache, interval):
    now = time.time()
    if not api_key:
        return "clear"
    if cache.get("last_time") and now - cache["last_time"] < interval:
        return cache.get("cond","clear")
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
        r = requests.get(url, timeout=5)
        d = r.json()
        w = d.get("weather", [])
        main = w[0]["main"].lower() if w else ""
        desc = w[0]["description"].lower() if w else ""
        vis = d.get("visibility", 10000)
        if "rain" in main or "drizzle" in main:
            rain_1h = d.get("rain", {}).get("1h", 0)
            cond = "heavy_rain" if "heavy" in desc or rain_1h > 5 else "light_rain"
        elif "snow" in main:
            cond = "snow"
        elif "mist" in main or "fog" in main or vis < 2000:
            cond = "fog"
        else:
            cond = "clear"
        cache["last_time"] = now
        cache["cond"] = cond
        return cond
    except Exception:
        return cache.get("cond","clear")

def build_training_dataframe(json_path):
    try:
        with open(json_path, "r") as f:
            arr = json.load(f)
    except Exception:
        return None
    rows = []
    for entry in arr:
        lc = entry.get("lane_counts", {})
        total = sum(lc.values())
        multipliers = entry.get("multipliers", {}) or {}
        weather_mult = multipliers.get("weather_mult", 1.0)
        rush_mult = multipliers.get("rush_mult", 1.0)
        short_mult = multipliers.get("short_term_mult", 1.0)
        decision = entry.get("decision", {})
        gt = decision.get("green_time")
        if gt is None:
            continue
        # use saved timestamp if present, else current hour
        hour = entry.get("timestamp_hour", datetime.now().hour)
        rows.append({
            "hour": hour,
            "total_vehicles": total,
            "weather_mult": float(weather_mult),
            "rush_mult": float(rush_mult),
            "short_mult": float(short_mult),
            "green_time": int(gt)
        })
    if not rows:
        return None
    return pd.DataFrame(rows)

def pretrain_incremental(df, scaler, model):
    X = df[["hour","total_vehicles","weather_mult","rush_mult","short_mult"]].values
    y = df["green_time"].values
    scaler.partial_fit(X)
    Xs = scaler.transform(X)
    # SGDRegressor requires at least one call to partial_fit with y shape
    model.partial_fit(Xs, y)
    return scaler, model

# ----------------- UI helpers for live display -----------------
def stream_frame_to_ui(frame, frame_placeholder, fps_placeholder, frame_idx, prev_time):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    now = time.time()
    fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
    fps_placeholder.markdown(f"**FPS:** {fps:.1f} | Frame: {frame_idx}")
    return now

# ----------------- Main detection (integrated) -----------------
def run_detection(
    video_path,
    use_weather,
    weather_key,
    weather_lat,
    weather_lon,
    pixels_per_meter,
    speed_limit_kmh,
    json_out_path,
    overspeed_out_path,
    heatmap_out_path,
    st_session_state,
    use_ml=False,
    ml_model=None,
    ml_scaler=None,
    online_learning=False,
    save_model_period_sec=300,
    save_model_path=ML_MODEL_FILE,
    save_scaler_path=ML_SCALER_FILE
):
    # load YOLO
    model_detect = YOLO("yolov8n.pt")
    cap, err = open_video_capture(video_path)
    if err:
        raise RuntimeError(err)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sliding_frames = int(SLIDING_WINDOW_SECONDS * fps)
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # initialize outputs
    try:
        with open(json_out_path, "w") as f:
            json.dump([], f)
    except Exception:
        pass

    overspeeding_vehicles = []
    vehicle_history = {}
    frame_number = 0
    hourly_counts = defaultdict(int)
    vehicle_count_window = deque()
    weather_cache = {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    frame_placeholder = st.empty()
    fps_placeholder = st.sidebar.empty()
    prev_time = time.time()
    last_save_time = prev_time

    while True:
        if st_session_state.get("stop_requested"):
            status_text.text("Stopped by user.")
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # detection using ultralytics (returns a Results object)
        results = model_detect(frame)[0]
        detections = []
        lane_counts = {"lane1":0, "lane2":0}
        ambulance_detected = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model_detect.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            if label in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1,y1,x2,y2,conf,label])
                if x1 < frame.shape[1]//2:
                    lane_counts["lane1"] += 1
                else:
                    lane_counts["lane2"] += 1
                if label == "ambulance":
                    ambulance_detected = True

        dets = np.array([[d[0],d[1],d[2],d[3],d[4]] for d in detections]) if detections else np.empty((0,5))
        tracks = tracker.update(dets)

        frame_data = []
        for tr in tracks:
            # track: x1,y1,x2,y2,track_id (float)
            x1,y1,x2,y2,tid = tr
            x1,y1,x2,y2,tid = int(x1), int(y1), int(x2), int(y2), int(tid)
            cx = (x1 + x2)//2
            cy = (y1 + y2)//2
            curr_center = (cx, cy)

            cls_name = "unknown"
            # best-effort match detection -> class
            for d in detections:
                bx1,by1,bx2,by2,conf,label = d
                if abs(bx1 - x1) < 20 and abs(by1 - y1) < 20:
                    cls_name = label
                    break

            speed_kmh = 0.0
            if tid in vehicle_history:
                prev = vehicle_history[tid]
                frames_diff = frame_number - prev["frame"]
                if frames_diff > 0:
                    speed_kmh = calculate_speed(prev["center"], curr_center, frames_diff, fps, pixels_per_meter)
                    if speed_kmh > speed_limit_kmh:
                        found = False
                        for v in overspeeding_vehicles:
                            if v["track_id"] == tid:
                                found = True
                                if speed_kmh > v["max_speed"]:
                                    v["max_speed"] = round(speed_kmh,2)
                                    v["last_frame"] = frame_number
                                break
                        if not found:
                            overspeeding_vehicles.append({
                                "track_id": tid,
                                "class": cls_name,
                                "max_speed": round(speed_kmh,2),
                                "speed_limit": speed_limit_kmh,
                                "first_detected_frame": frame_number,
                                "last_frame": frame_number
                            })
            vehicle_history[tid] = {"center": curr_center, "frame": frame_number, "speed": speed_kmh}

            box_h = max(1, y2 - y1)
            dist_est = DISTANCE_FACTOR / box_h
            frame_data.append({
                "frame": frame_number,
                "track_id": tid,
                "class": cls_name,
                "distance": round(dist_est,2),
                "speed_kmh": round(speed_kmh,2),
                "is_overspeeding": speed_kmh > speed_limit_kmh
            })

        # cleanup old history
        if frame_number % 200 == 0:
            vehicle_history = {k:v for k,v in vehicle_history.items() if frame_number - v["frame"] < 300}

        # multipliers & sliding window congestion
        total_vehicles = sum(lane_counts.values())
        vehicle_count_window.append((frame_number, total_vehicles))
        while vehicle_count_window and (frame_number - vehicle_count_window[0][0]) > sliding_frames:
            vehicle_count_window.popleft()
        avg_recent = (sum(v for _,v in vehicle_count_window)/len(vehicle_count_window)) if vehicle_count_window else 0
        if avg_recent > 15:
            short_term_level = "high"
        elif avg_recent > 5:
            short_term_level = "moderate"
        else:
            short_term_level = "low"
        short_term_mult = SHORT_TERM_CONGESTION_MULTIPLIER[short_term_level]

        now_hour = datetime.now().hour
        hourly_counts[now_hour] += total_vehicles

        counts_list = list(hourly_counts.values())
        if counts_list:
            sorted_counts = sorted(counts_list, reverse=True)
            idx = max(0, int(len(sorted_counts)*0.2)-1)
            threshold = sorted_counts[idx] if sorted_counts else 0
            rush_mult = RUSH_HOUR_MULTIPLIER_PEAK if hourly_counts[now_hour] >= threshold and hourly_counts[now_hour] > 0 else 1.0
        else:
            rush_mult = 1.0

        # weather
        cond = "clear"
        weather_mult = 1.0
        if use_weather:
            cond = fetch_weather_condition_cached(weather_key, weather_lat, weather_lon, weather_cache, WEATHER_POLL_INTERVAL_SEC)
            weather_mult = WEATHER_GREEN_EXTENSION_FACTOR.get(cond, 1.0)

        # rule-based green_time
        if ambulance_detected:
            rule_base_green = 10
            rule_reason = "Ambulance override"
        else:
            if total_vehicles > 15:
                base = 20
                reason = "High congestion"
            elif total_vehicles > 5:
                base = 10
                reason = "Moderate congestion"
            else:
                base = 5
                reason = "Low traffic"
            rule_base_green = min(int(round(base * weather_mult * rush_mult * short_term_mult)), ML_MAX_GREEN)
            rule_reason = reason

        # ML prediction
        predicted_green = None
        if use_ml and ml_model is not None and ml_scaler is not None:
            feat = np.array([[now_hour, total_vehicles, weather_mult, rush_mult, short_term_mult]])
            try:
                feat_s = ml_scaler.transform(feat)
                p = ml_model.predict(feat_s)[0]
                p = max(ML_MIN_GREEN, min(ML_MAX_GREEN, int(round(float(p)))))
                predicted_green = p
            except Exception:
                predicted_green = None

        if predicted_green is not None:
            decision = {"signal":"GREEN", "green_time": predicted_green, "reason": f"ML-pred (rule:{rule_base_green}s)"}
        else:
            decision = {"signal":"GREEN", "green_time": rule_base_green, "reason": rule_reason}

        # append to traffic_output.json
        try:
            with open(json_out_path, "r+") as jf:
                try:
                    arr = json.load(jf)
                except Exception:
                    arr = []
                arr.append({
                    "timestamp": datetime.now().isoformat(),
                    "timestamp_hour": now_hour,
                    "frame": frame_number,
                    "vehicles": frame_data,
                    "lane_counts": lane_counts,
                    "decision": decision,
                    "overspeeding_count": sum(1 for v in frame_data if v.get("is_overspeeding")),
                    "weather": cond,
                    "multipliers": {"weather_mult":weather_mult, "rush_mult":rush_mult, "short_term_mult":short_term_mult},
                    "ml_used": (predicted_green is not None)
                })
                jf.seek(0)
                json.dump(arr, jf, indent=2)
        except Exception:
            pass

        # online learning update (use rule_base_green as supervised label)
        if online_learning and ml_model is not None and ml_scaler is not None:
            X_new = np.array([[now_hour, total_vehicles, weather_mult, rush_mult, short_term_mult]])
            try:
                ml_scaler.partial_fit(X_new)
                Xs_new = ml_scaler.transform(X_new)
                y_new = np.array([rule_base_green])
                ml_model.partial_fit(Xs_new, y_new)
            except Exception:
                pass

        # periodic save
        if online_learning and (time.time() - last_save_time) > save_model_period_sec:
            try:
                joblib.dump(ml_model, save_model_path)
                joblib.dump(ml_scaler, save_scaler_path)
                last_save_time = time.time()
            except Exception:
                pass

        # draw bounding boxes + ids on frame for UI
        for tr in tracks:
            x1,y1,x2,y2,tid = tr
            x1,y1,x2,y2,tid = int(x1),int(y1),int(x2),int(y2),int(tid)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # display live frame
        prev_time = stream_frame_to_ui(frame, frame_placeholder, fps_placeholder, frame_number, prev_time)

        # progress/status
        if total_frames:
            progress_bar.progress(min(1.0, frame_number / total_frames))
        status_text.text(f"Frame {frame_number} | Vehicles: {total_vehicles} | Decision: {decision['green_time']}s | Weather: {cond} | ML_used: {predicted_green is not None}")

        # safety limit (avoid infinite runs)
        # (you can remove or increase this)
        if total_frames and frame_number >= total_frames:
            break

    # end loop
    cap.release()

    # write overspeeding & heatmap files
    try:
        with open(overspeed_out_path, "w") as of:
            json.dump({"speed_limit_kmh": speed_limit_kmh, "total_overspeeding_vehicles": len(overspeeding_vehicles), "vehicles": overspeeding_vehicles}, of, indent=2)
    except Exception:
        pass
    try:
        with open(heatmap_out_path, "w") as hf:
            json.dump(dict(hourly_counts), hf, indent=2)
    except Exception:
        pass

    # final model save
    if online_learning and ml_model is not None and ml_scaler is not None:
        try:
            joblib.dump(ml_model, save_model_path)
            joblib.dump(ml_scaler, save_scaler_path)
        except Exception:
            pass

    return {"frames": frame_number, "overspeeding_count": len(overspeeding_vehicles), "heatmap": dict(hourly_counts), "overspeeding_list": overspeeding_vehicles}

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Traffic Demo — Streamlit UI (online-ML)", layout="wide")
st.title("Traffic Detection & Smart Signal — Streamlit GUI (Online ML)")

col1, col2 = st.columns([2,1])

with col1:
    st.header("Run Detection")
    video_path = st.text_input("Video path", DEFAULT_VIDEO)
    use_weather = st.checkbox("Enable weather-aware timing", value=True)
    weather_key = st.text_input("OpenWeatherMap API key (leave empty to disable)", value=os.getenv("OPENWEATHERMAP_API_KEY",""))
    weather_lat = st.number_input("Weather latitude", value=28.6139, format="%.6f")
    weather_lon = st.number_input("Weather longitude", value=77.2090, format="%.6f")
    pixels_per_meter = st.number_input("Pixels per meter (calibration)", value=PIXELS_PER_METER, min_value=1)
    speed_limit_kmh = st.number_input("Speed limit (km/h)", value=SPEED_LIMIT_KMH, min_value=10)

    st.markdown("**ML options (online learning)**")
    enable_ml = st.checkbox("Enable ML features (load/train model)", value=True)
    online_learning = st.checkbox("Enable online learning (update model during detection)", value=True)
    save_model_period_sec = st.number_input("Model save interval (seconds)", value=300, min_value=30)
    load_existing_model = st.checkbox("Load existing saved model from disk (if available)", value=True)
    use_ml_for_run = st.checkbox("Use ML prediction while running detection (if model available)", value=True)

    run_col1, run_col2 = st.columns(2)
    run_btn = run_col1.button("Run Detection")
    stop_btn = run_col2.button("Stop")
    if "stop_requested" not in st.session_state:
        st.session_state["stop_requested"] = False
    if stop_btn:
        st.session_state["stop_requested"] = True

with col2:
    st.header("Model / Exports")
    st.markdown("Model file: `ml_model.joblib`, scaler: `ml_scaler.joblib`")

    ml_model = None
    ml_scaler = None
    train_df = build_training_dataframe(JSON_FILE)
    if load_existing_model and os.path.exists(ML_MODEL_FILE) and os.path.exists(ML_SCALER_FILE) and enable_ml:
        try:
            ml_model = joblib.load(ML_MODEL_FILE)
            ml_scaler = joblib.load(ML_SCALER_FILE)
            st.success("Loaded saved ML model and scaler from disk.")
        except Exception as e:
            st.warning(f"Failed to load saved model: {e}")

    if ml_model is None and enable_ml:
        if train_df is not None and len(train_df) >= MIN_ROWS_TO_TRAIN:
            ml_scaler = StandardScaler()
            ml_model = SGDRegressor(max_iter=1000, tol=1e-3)
            try:
                ml_scaler, ml_model = pretrain_incremental(train_df, ml_scaler, ml_model)
                st.success(f"Pretrained incremental ML model on {len(train_df)} rows.")
                X = train_df[["hour","total_vehicles","weather_mult","rush_mult","short_mult"]].values
                y = train_df["green_time"].values
                Xs = ml_scaler.transform(X)
                Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
                preds = ml_model.predict(Xte)
                r2 = r2_score(yte, preds)
                mae = mean_absolute_error(yte, preds)
                st.write(f"Pretrain eval — R²: {r2:.3f}, MAE: {mae:.2f}")
            except Exception as e:
                st.warning(f"Pretraining failed: {e}")
        else:
            st.info(f"Not enough historic rows in {JSON_FILE} to pretrain (need ≥ {MIN_ROWS_TO_TRAIN}). A model will still be created and trained online during detection.")

    # Overspeeding exports and heatmap display
    st.subheader("Overspeeding Exports")
    try:
        with open(OVERSPEEDING_FILE, "r") as f:
            over_obj = json.load(f)
            overspeed_df = pd.DataFrame(over_obj.get("vehicles", []))
    except Exception:
        try:
            with open(JSON_FILE, "r") as jf:
                data_loaded = json.load(jf)
        except Exception:
            data_loaded = []
        rows = []
        for entry in data_loaded:
            for v in entry.get("vehicles", []):
                if v.get("is_overspeeding"):
                    rows.append({"frame":entry.get("frame"), **v})
        overspeed_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if not overspeed_df.empty:
        st.dataframe(overspeed_df)
        csv_buf = overspeed_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_buf, file_name="overspeeding_vehicles.csv", mime="text/csv")
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            overspeed_df.to_excel(writer, index=False, sheet_name="overspeeding")
        st.download_button("Download Excel", data=excel_buffer.getvalue(), file_name="overspeeding_vehicles.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("No overspeeding data available yet.")

    st.subheader("Hourly Heatmap")
    try:
        with open(HEATMAP_FILE, "r") as hf:
            heatmap = json.load(hf)
    except Exception:
        heatmap = {}
    if heatmap:
        hm_series = pd.Series(heatmap).sort_index()
        hm_df = hm_series.reindex(range(24), fill_value=0).rename("counts").reset_index().rename(columns={"index":"hour"})
        st.bar_chart(data=hm_df.set_index("hour"))
    else:
        st.info("No heatmap data yet (run detection to generate).")

# ----------------- Run detection on button press -----------------
if run_btn:
    st.session_state["stop_requested"] = False
    try:
        # prepare ML for online learning if enabled
        if enable_ml and (ml_model is None or ml_scaler is None):
            ml_scaler = StandardScaler()
            ml_model = SGDRegressor(max_iter=1000, tol=1e-3)

        res = run_detection(
            video_path,
            use_weather,
            weather_key,
            weather_lat,
            weather_lon,
            pixels_per_meter,
            speed_limit_kmh,
            JSON_FILE,
            OVERSPEEDING_FILE,
            HEATMAP_FILE,
            st.session_state,
            use_ml = use_ml_for_run and enable_ml and (ml_model is not None),
            ml_model = ml_model,
            ml_scaler = ml_scaler,
            online_learning = online_learning and enable_ml,
            save_model_period_sec = int(save_model_period_sec),
            save_model_path = ML_MODEL_FILE,
            save_scaler_path = ML_SCALER_FILE
        )

        st.success(f"Processing finished — frames: {res.get('frames',0)}, overspeeding vehicles: {res.get('overspeeding_count',0)}")
        if os.path.exists(ML_MODEL_FILE) and os.path.exists(ML_SCALER_FILE):
            with open(ML_MODEL_FILE, "rb") as mm:
                st.download_button("Download ML model (joblib)", data=mm, file_name=ML_MODEL_FILE, mime="application/octet-stream")
            with open(ML_SCALER_FILE, "rb") as ms:
                st.download_button("Download ML scaler (joblib)", data=ms, file_name=ML_SCALER_FILE, mime="application/octet-stream")
    except Exception as e:
        st.error(f"Error during processing: {e}")
