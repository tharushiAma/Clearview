# Run ClearView Python Backend
Write-Host "==============================" -ForegroundColor Yellow
Write-Host "Starting Python Backend Server" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Yellow
Set-Location $PSScriptRoot
# Execute the script from the backend sub-folder
python backend/backend_server.py
