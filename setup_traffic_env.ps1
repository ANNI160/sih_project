<#
Setup script: creates venv at E:\traffic_env, installs dependencies, optionally runs Streamlit.
Usage:
  - Save as setup_traffic_env.ps1
  - Run: PowerShell -ExecutionPolicy Bypass -File .\setup_traffic_env.ps1
  - To install and immediately run the app: PowerShell -ExecutionPolicy Bypass -File .\setup_traffic_env.ps1 -RunApp
#>

param(
    [switch]$RunApp
)

# CONFIG
$venvPath = "E:\traffic_env"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$pipExe = Join-Path $venvPath "Scripts\pip.exe"
$requirements = @(
    "streamlit",
    "ultralytics",
    "opencv-python-headless",
    "pandas",
    "numpy",
    "scikit-learn",
    "openpyxl",
    "requests",
    "joblib",
    "pyarrow==15.0.0"
)

Write-Host "== Traffic env setup script ==" -ForegroundColor Cyan
Write-Host "Virtual env path: $venvPath`n"

# 1) Create venv if it doesn't exist
if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath ..." -ForegroundColor Yellow
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create venv. Ensure a system Python (python) is on PATH and retry."
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists at $venvPath. Skipping creation." -ForegroundColor Yellow
}

# 2) Use venv python to upgrade pip
if (-Not (Test-Path $pythonExe)) {
    Write-Error "Cannot find venv python executable at $pythonExe. Aborting."
    exit 1
}

Write-Host "Upgrading pip inside venv..." -ForegroundColor Yellow
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Warning "pip upgrade returned non-zero exit code. Continuing anyway."
}

# 3) Install requirements (use venv pip)
Write-Host "Installing packages: $($requirements -join ', ')" -ForegroundColor Yellow
foreach ($pkg in $requirements) {
    Write-Host "-> Installing $pkg ..." -NoNewline
    & $pythonExe -m pip install $pkg
    if ($LASTEXITCODE -eq 0) {
        Write-Host " Done" -ForegroundColor Green
    } else {
        Write-Host " Failed (continuing)" -ForegroundColor Red
    }
}

# 4) Verify pyarrow
Write-Host "`nVerifying pyarrow installation..." -ForegroundColor Cyan
try {
    $ver = & $pythonExe -c "import pyarrow as pa; print(pa.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "pyarrow version: $ver" -ForegroundColor Green
    } else {
        Write-Warning "pyarrow import failed or printed errors:`n$ver"
    }
} catch {
    Write-Warning "pyarrow verification threw an exception: $_"
}

# 5) Summary / instructions
Write-Host "`nSetup finished." -ForegroundColor Cyan
Write-Host "To activate the venv in PowerShell for interactive use, run:"
Write-Host "`tSet-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force" -ForegroundColor Yellow
Write-Host "`t& `"$venvPath\Scripts\Activate.ps1`"" -ForegroundColor Yellow
Write-Host "Or use the venv python directly: `"$pythonExe`"" -ForegroundColor Yellow

# 6) Optionally run the Streamlit app
if ($RunApp) {
    Write-Host "`nLaunching Streamlit app (this will block the terminal)..." -ForegroundColor Cyan
    # Launch using the venv python to avoid PATH issues
    & $pythonExe -m streamlit run traffic_demo_streamlit.py
} else {
    Write-Host "`nIf you want to run the app now, run:" -ForegroundColor Cyan
    Write-Host "`tPowerShell -ExecutionPolicy Bypass -File .\setup_traffic_env.ps1 -RunApp" -ForegroundColor Yellow
    Write-Host "or after activating the venv run:" -ForegroundColor Cyan
    Write-Host "`tstreamlit run traffic_demo_streamlit.py" -ForegroundColor Yellow
}
