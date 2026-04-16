# Run ClearView Full Stack Application
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Starting ClearView Full Stack Application" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will start TWO servers:" -ForegroundColor Yellow
Write-Host "  1. Python FastAPI Backend  (port 8000)  - from website/backend_server.py" -ForegroundColor Cyan
Write-Host "  2. Next.js Frontend        (port 3000)  - from website/frontend/" -ForegroundColor Cyan
Write-Host ""

# Start backend in a new terminal window
Write-Host "Starting Python backend server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python backend/backend_server.py"

# Wait for backend to initialise (model loading takes ~20-30s on first startup)
Write-Host "Waiting for backend to initialise (30 seconds for model loading)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Start Next.js frontend from the frontend sub-directory
Write-Host "Starting Next.js frontend server..." -ForegroundColor Green
Write-Host ""
Set-Location "$PSScriptRoot\frontend"
cmd /c "npm run dev"
