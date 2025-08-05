# PowerShell script to run both simulator and predictor
# Usage: .\run-all.ps1 [number_of_games]

param(
    [int]$NumGames = 10000
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   21-Point Game AI - Complete Pipeline" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if dotnet is installed
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    Write-Host "Error: .NET SDK is not installed" -ForegroundColor Red
    exit 1
}

# Step 1: Run simulator
Write-Host "STEP 1: Running Game Simulator" -ForegroundColor Yellow
Write-Host "-------------------------------" -ForegroundColor Yellow
.\run-simulator.ps1 -NumGames $NumGames

if ($LASTEXITCODE -ne 0) {
    Write-Host "Simulator failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host ""

# Step 2: Run predictor
Write-Host "STEP 2: Running Action Predictor" -ForegroundColor Yellow
Write-Host "---------------------------------" -ForegroundColor Yellow
.\run-predictor.ps1

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "          Pipeline Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green