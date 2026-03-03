# Run ClearView Python Backend
Write-Host "==============================" -ForegroundColor Yellow
Write-Host "Starting Python Backend Server" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Yellow
Set-Location $PSScriptRoot
python backend_server.py
