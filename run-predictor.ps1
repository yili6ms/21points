# PowerShell script to run the Action Predictor
# Usage: .\run-predictor.ps1 [csv_file]           - Train mode
#        .\run-predictor.ps1 -Test                - Test mode (interactive)
#        .\run-predictor.ps1 -Test -ModelFile model.json -TestFile test_data.csv - Test mode (direct)
# Default: gameplay_log.csv

param(
    [string]$InputFile = "gameplay_log.csv",
    [switch]$Test,
    [string]$ModelFile = "",
    [string]$TestFile = ""
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "     21-Point Action Predictor" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if dotnet is installed
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    Write-Host "Error: .NET SDK is not installed" -ForegroundColor Red
    exit 1
}

# Build the project
Write-Host "Building ActionPredictor..." -ForegroundColor Yellow
dotnet build ActionPredictor/ActionPredictor.fsproj --nologo -v quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Check if this is test mode
if ($Test) {
    if ($ModelFile -and $TestFile) {
        # Direct test mode with model and test file
        Write-Host "Testing model: $ModelFile" -ForegroundColor Yellow
        Write-Host "Test data: $TestFile" -ForegroundColor Yellow
        Write-Host ""
        dotnet run --project ActionPredictor/ActionPredictor.fsproj --no-build -- --test $ModelFile $TestFile
    } else {
        # Interactive test mode
        Write-Host "Interactive test mode" -ForegroundColor Yellow
        Write-Host ""
        dotnet run --project ActionPredictor/ActionPredictor.fsproj --no-build -- --test
    }
} else {
    # Training mode
    # Check if input file exists
    if (-not (Test-Path $InputFile)) {
        Write-Host "Error: Input file '$InputFile' not found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please run the simulator first:" -ForegroundColor Yellow
        Write-Host "  .\run-simulator.ps1" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "Training mode" -ForegroundColor Yellow
    Write-Host "Using input file: $InputFile" -ForegroundColor Yellow
    Write-Host ""
    
    # Run the predictor in training mode
    dotnet run --project ActionPredictor/ActionPredictor.fsproj --no-build -- $InputFile
}