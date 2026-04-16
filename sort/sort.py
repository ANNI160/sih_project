# sort.py
# Minimal SORT implementation compatible with traffic_demo.py
# Requires: numpy, filterpy, scipy

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import math
import time

def iou(bb_test, bb_gt):
    """
    Computes IOU between two bounding boxes in [x1,y1,x2,y2] format.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box [x1,y1,x2,y2] and returns z in the form
    [center_x, center_y, s, r] where s = scale/area, r = aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.
    cy = bbox[1] + h / 2.
    s = w * h
    r = 0.
    if h > 0:
        r = w / float(h)
    return np.array([cx, cy, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a state vector and returns bbox [x1,y1,x2,y2]
    x: state vector [cx, cy, s, r, vx, vy, vs, vr] possibly
    """
    cx = x[0]
    cy = x[1]
    s = x[2]
    r = x[3]
    if s <= 0 or r <= 0:
        w = 0
        h = 0
    else:
        w = math.sqrt(s * r)
        h = s / w if w != 0 else 0
    x1 = cx - w / 2.
    y1 = cy - h / 2.
    x2 = cx + w / 2.
    y2 = cy + h / 2.
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))

class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as bounding boxes.
    """
    count = 0

    def __init__(self, bbox, dt=1.0):
        # bbox: [x1,y1,x2,y2]
        # Initialize a 8D Kalman filter [cx,cy,s,r, vx,vy,vs,vr]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # State transition
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i+4] = dt

        # Measurement function maps state to [cx,cy,s,r]
        self.kf.H = np.zeros((4, 8))
        self.kf.H[0, 0] = 1.0
        self.kf.H[1, 1] = 1.0
        self.kf.H[2, 2] = 1.0
        self.kf.H[3, 3] = 1.0

        # Reasonable uncertainty
        self.kf.R[0:4, 0:4] *= 10.
        self.kf.P[0:4, 0:4] *= 10.
        self.kf.P[4:8, 4:8] *= 1000.  # Give high uncertainty to the velocities

        # Process noise
        q = 1.0
        self.kf.Q[0:4, 0:4] *= q
        self.kf.Q[4:8, 4:8] *= q

        z = convert_bbox_to_z(bbox)
        self.kf.x[:4] = z.reshape((4, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count + 1
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_update_time = time.time()

    def update(self, bbox):
        # bbox: [x1,y1,x2,y2]
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        z = convert_bbox_to_z(bbox)
        self.kf.update(z)
        self.last_update_time = time.time()

    def predict(self):
        # Advances the state vector and returns the predicted bounding box estimate.
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        # Returns the current bounding box estimate [x1,y1,x2,y2]
        return convert_x_to_bbox(self.kf.x).reshape((4,))

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        max_age: frames to keep alive without updates
        min_hits: number of hits needed to consider track confirmed
        iou_threshold: matching threshold
        """
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.trackers = []

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - numpy array of detections in the format [[x1,y1,x2,y2,score], ...]
        Returns:
          np.array with shape N x 5 containing tracked objects as [x1,y1,x2,y2,track_id]
        """
        # Predict all trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict().reshape((4,))
            trks[t, :4] = pos
            trks[t, 4] = trk.id
            if np.any(np.isnan(pos)):
                to_del.append(t)
        # Remove dead predictors
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        dets_array = dets.copy() if dets is not None else np.empty((0,5))
        if dets_array.shape[0] == 0:
            dets_array = np.empty((0,5))

        # If there are trackers and detections, compute IOU cost matrix
        if len(self.trackers) > 0 and dets_array.shape[0] > 0:
            iou_matrix = np.zeros((len(self.trackers), dets_array.shape[0]), dtype=np.float32)
            for t, trk in enumerate(self.trackers):
                for d in range(dets_array.shape[0]):
                    iou_matrix[t, d] = iou(trks[t, :4], dets_array[d, :4])

            # Solve assignment (maximize IOU -> minimize -IOU)
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(matched_indices).T

            unmatched_trks = list(range(len(self.trackers)))
            unmatched_dets = list(range(dets_array.shape[0]))
            matches = []

            for m in matched_indices:
                t, d = m[0], m[1]
                if iou_matrix[t, d] < self.iou_threshold:
                    continue
                matches.append((t, d))
                unmatched_trks.remove(t)
                unmatched_dets.remove(d)
        else:
            matches = []
            unmatched_trks = list(range(len(self.trackers)))
            unmatched_dets = list(range(dets_array.shape[0]))

        # Update matched trackers with assigned detections
        for t, d in matches:
            self.trackers[t].update(dets_array[d, :4])

        # Create and initialize new trackers for unmatched detections
        for idx in unmatched_dets:
            trk = KalmanBoxTracker(dets_array[idx, :4])
            self.trackers.append(trk)

        i = len(self.trackers)
        # Prepare return list and remove dead trackers
        removed = []
        outputs = []
        for trk in reversed(self.trackers):
            d = trk.get_state()
            tid = trk.id
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.min_hits == 0):
                outputs.append(np.array([d[0], d[1], d[2], d[3], tid]))
            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                removed.append(trk)
                self.trackers.pop(i)

        if len(outputs) > 0:
            outputs = np.vstack(outputs)
        else:
            outputs = np.empty((0, 5))

        # Convert to np.array with ints for bbox coords (so consumer can map to image indices)
        # but keep track id as int
        # outputs: [x1,y1,x2,y2,id]
        out = []
        for row in outputs:
            x1, y1, x2, y2, tid = row
            out.append([float(x1), float(y1), float(x2), float(y2), int(tid)])
        if len(out) > 0:
            return np.array(out)
        else:
            return np.empty((0, 5))
