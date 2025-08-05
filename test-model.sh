#!/bin/bash

# Test a trained model with new data
# Usage: ./test-model.sh [model_file.json] [test_data.csv]

echo "========================================="
echo "        21-Point Model Tester"
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

# If both model and test file are provided
if [ $# -eq 2 ]; then
    MODEL_FILE="$1"
    TEST_FILE="$2"
    
    if [ ! -f "$MODEL_FILE" ]; then
        echo "Error: Model file '$MODEL_FILE' not found!"
        exit 1
    fi
    
    if [ ! -f "$TEST_FILE" ]; then
        echo "Error: Test data file '$TEST_FILE' not found!"
        echo ""
        echo "Generate new test data with:"
        echo "  ./run-simulator.sh 1000"
        exit 1
    fi
    
    echo "Testing model: $MODEL_FILE"
    echo "Test data: $TEST_FILE"
    echo ""
    
    dotnet run --project ActionPredictor/ActionPredictor.fsproj --no-build -- --test "$MODEL_FILE" "$TEST_FILE"
else
    # Interactive mode
    echo "Interactive mode - will list available models"
    echo ""
    
    dotnet run --project ActionPredictor/ActionPredictor.fsproj --no-build -- --test
fi