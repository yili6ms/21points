# PowerShell script to test a trained model
# Usage: .\test-model.ps1 [model_file.json] [test_data.csv]

param(
    [string]$ModelFile = "",
    [string]$TestFile = ""
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "        21-Point Model Tester" -ForegroundColor Cyan
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

# If both model and test file are provided
if ($ModelFile -and $TestFile) {
    if (-not (Test-Path $ModelFile)) {
        Write-Host "Error: Model file '$ModelFile' not found!" -ForegroundColor Red
        exit 1
    }
    
    if (-not (Test-Path $TestFile)) {
        Write-Host "Error: Test data file '$TestFile' not found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Generate new test data with:" -ForegroundColor Yellow
        Write-Host "  .\run-simulator.ps1 -NumGames 1000" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "Testing model: $ModelFile" -ForegroundColor Yellow
    Write-Host "Test data: $TestFile" -ForegroundColor Yellow
    Write-Host ""
    
    dotnet run --project ActionPredictor/ActionPredictor.fsproj --no-build -- --test $ModelFile $TestFile
} else {
    # Interactive mode
    Write-Host "Interactive mode - will list available models" -ForegroundColor Yellow
    Write-Host ""
    
    dotnet run --project ActionPredictor/ActionPredictor.fsproj --no-build -- --test
}