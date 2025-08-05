#!/bin/bash

# Run the Action Predictor
# Usage: ./run-predictor.sh [csv_file]           - Train mode
#        ./run-predictor.sh --test              - Test mode (interactive)
#        ./run-predictor.sh --test model.json test_data.csv - Test mode (direct)
# Default: gameplay_log.csv

echo "========================================="
echo "     21-Point Action Predictor"
echo "========================================="

# Check if dotnet is installed
if ! command -v dotnet &> /dev/null; then
    echo "Error: .NET SDK is not installed"
    exit 1
fi

# Build the project
echo "Building ActionPredictor..."
dotnet build ActionPredictor/ActionPredictor.fsproj --nologo -v quiet

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Check if this is test mode
if [ "$1" = "--test" ]; then
    if [ $# -eq 3 ]; then
        # Direct test mode with model and test file
        MODEL_FILE="$2"
        TEST_FILE="$3"
        echo "Testing model: $MODEL_FILE"
        echo "Test data: $TEST_FILE"
        echo ""
        dotnet run --project ActionPredictor/ActionPredictor.fsproj --no-build -- --test "$MODEL_FILE" "$TEST_FILE"
    else
        # Interactive test mode
        echo "Interactive test mode"
        echo ""
        dotnet run --project ActionPredictor/ActionPredictor.fsproj --no-build -- --test
    fi
else
    # Training mode
    INPUT_FILE=${1:-gameplay_log.csv}
    
    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Input file '$INPUT_FILE' not found!"
        echo ""
        echo "Please run the simulator first:"
        echo "  ./run-simulator.sh"
        exit 1
    fi
    
    echo "Training mode"
    echo "Using input file: $INPUT_FILE"
    echo ""
    
    # Run the predictor in training mode
    dotnet run --project ActionPredictor/ActionPredictor.fsproj --no-build -- "$INPUT_FILE"
fi