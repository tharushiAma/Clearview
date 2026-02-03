# Start the ClearView Backend
Write-Host "Starting ClearView Backend..." -ForegroundColor Green

# Check for Python
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python not found! Please install Python 3.10+" -ForegroundColor Red
    exit
}

# Install dependencies if needed
if (Test-Path "requirements.txt") {
    Write-Host "Checking dependencies..."
    pip install -r requirements.txt
}

# Run Server
Write-Host "Starting Uvicorn Server..."
python backend_server.py
