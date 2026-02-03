# ClearView Backend Server Startup Script

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Starting ClearView Python Backend Server" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Define Python path
$pythonPath = "c:\Users\lucif\Desktop\Clearview\.venv\Scripts\python.exe"

# Check if Python exists
if (-not (Test-Path $pythonPath)) {
    Write-Host "Error: Python not found at $pythonPath" -ForegroundColor Red
    Write-Host "Please check your virtual environment path." -ForegroundColor Yellow
    exit 1
}

Write-Host "Python: $pythonPath" -ForegroundColor Cyan
Write-Host "Server will run on: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Loading ML models... (this may take 30-60 seconds on first run)" -ForegroundColor Yellow
Write-Host ""

# Install FastAPI and uvicorn if not already installed
Write-Host "Checking dependencies..." -ForegroundColor Cyan
& $pythonPath -m pip install fastapi uvicorn --quiet

# Start the backend server
Write-Host "Starting server..." -ForegroundColor Green
& $pythonPath backend_server.py

# This script will keep running until you press Ctrl+C
