#!/bin/bash

# Run the Game Simulator
# Usage: ./run-simulator.sh [number_of_games]
# Default: 10000 games

echo "========================================="
echo "     21-Point Game Simulator"
echo "========================================="

# Check if dotnet is installed
if ! command -v dotnet &> /dev/null; then
    echo "Error: .NET SDK is not installed"
    exit 1
fi

# Number of games (default 10000 if not specified)
NUM_GAMES=${1:-10000}

echo "Simulating $NUM_GAMES games..."
echo ""

# Build the project
echo "Building GameSimulator..."
dotnet build GameSimulator/GameSimulator.fsproj --nologo -v quiet

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Run the simulator
dotnet run --project GameSimulator/GameSimulator.fsproj --no-build -- $NUM_GAMES

# Check if files were created
if [ -f "gameplay_log.csv" ]; then
    echo ""
    echo "Output files created:"
    echo "  - gameplay_log.csv ($(wc -l < gameplay_log.csv) lines)"
    echo "  - gameplay_log.json"
else
    echo "Warning: Output files were not created"
fi