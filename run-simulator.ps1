# PowerShell script to run the Game Simulator
# Usage: .\run-simulator.ps1 [number_of_games]
# Default: 10000 games

param(
    [int]$NumGames = 10000
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "     21-Point Game Simulator" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if dotnet is installed
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    Write-Host "Error: .NET SDK is not installed" -ForegroundColor Red
    exit 1
}

Write-Host "Simulating $NumGames games..." -ForegroundColor Yellow
Write-Host ""

# Build the project
Write-Host "Building GameSimulator..." -ForegroundColor Yellow
dotnet build GameSimulator/GameSimulator.fsproj --nologo -v quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Run the simulator
dotnet run --project GameSimulator/GameSimulator.fsproj --no-build -- $NumGames

# Check if files were created
if (Test-Path "gameplay_log.csv") {
    $lineCount = (Get-Content "gameplay_log.csv" | Measure-Object -Line).Lines
    Write-Host ""
    Write-Host "Output files created:" -ForegroundColor Green
    Write-Host "  - gameplay_log.csv ($lineCount lines)" -ForegroundColor Green
    Write-Host "  - gameplay_log.json" -ForegroundColor Green
} else {
    Write-Host "Warning: Output files were not created" -ForegroundColor Yellow
}