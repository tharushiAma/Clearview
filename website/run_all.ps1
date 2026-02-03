# Start Both Servers - ClearView Application

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Starting ClearView Full Stack Application" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This will start TWO servers:" -ForegroundColor Yellow
Write-Host "  1. Python FastAPI Backend (port 8000)" -ForegroundColor Cyan
Write-Host "  2. Next.js Frontend (port 3000)" -ForegroundColor Cyan
Write-Host ""

# Start backend in a new terminal
Write-Host "Starting Python backend server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-File", ".\run_backend.ps1"

# Wait a moment for backend to start
Write-Host "Waiting for backend to initialize (5 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start Next.js frontend in current terminal
Write-Host "Starting Next.js frontend server..." -ForegroundColor Green
Write-Host ""
npm run dev
