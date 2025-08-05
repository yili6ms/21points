#!/bin/bash

# Run both the simulator and predictor in sequence
# Usage: ./run-all.sh [number_of_games]

echo "========================================="
echo "   21-Point Game AI - Complete Pipeline"
echo "========================================="
echo ""

# Check if dotnet is installed
if ! command -v dotnet &> /dev/null; then
    echo "Error: .NET SDK is not installed"
    exit 1
fi

NUM_GAMES=${1:-10000}

# Step 1: Run simulator
echo "STEP 1: Running Game Simulator"
echo "-------------------------------"
./run-simulator.sh $NUM_GAMES

if [ $? -ne 0 ]; then
    echo "Simulator failed!"
    exit 1
fi

echo ""
echo ""

# Step 2: Run predictor
echo "STEP 2: Running Action Predictor"
echo "---------------------------------"
./run-predictor.sh

echo ""
echo "========================================="
echo "          Pipeline Complete!"
echo "========================================="