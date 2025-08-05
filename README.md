# 21-Point Card Game AI

A comprehensive F# implementation of AI strategies for the 21-point card game using multiple machine learning approaches including traditional neural networks and ResNet architectures.

## Overview

This project implements an AI system that learns optimal Hit/Stand strategies for a simplified 21-point card game through gameplay simulation and neural network training. The system consists of two main components:

1. **GameSimulator** - Simulates thousands of games and generates training data
2. **ActionPredictor** - Trains various AI models to predict optimal actions

## Game Rules

- Start with a score of 0
- Draw cards with values 1-10 (uniform random distribution)
- Goal: Reach as close to 21 as possible without going over
- Actions: **Hit** (draw another card) or **Stand** (keep current score)
- Bust if score > 21

## Architecture

### Project Structure
```
21points/
├── GameSimulator/          # Game simulation and data generation
│   ├── Program.fs          # Main simulation orchestrator
│   ├── Game.fs             # Core game logic
│   ├── CardDeck.fs         # Card drawing mechanics
│   ├── Player.fs           # Player strategy implementation
│   ├── PlayLogger.fs       # Data logging to CSV/JSON
│   └── Types.fs            # Shared data types
├── ActionPredictor/        # AI training and prediction
│   ├── Program.fs          # Training method selection
│   ├── DataLoader.fs       # CSV data loading
│   ├── FeatureExtractor.fs # Feature engineering (12 features)
│   ├── StandardTrainer.fs  # Rule-based policy learning
│   ├── NeuralNetwork.fs    # Simple neural network (12→32→32→2)
│   ├── NeuralNetworkTrainer.fs # Standard NN training
│   ├── ResNet.fs           # ResNet architecture with skip connections
│   ├── ResNetTrainer.fs    # Advanced ResNet training
│   ├── ModelPersistence.fs # Model saving/loading
│   ├── ModelTester.fs      # Model evaluation
│   └── Types.fs            # Shared data types
├── gameplay_log.csv        # Generated training data
└── *.pt                    # Saved neural network models
```

### Data Types

```fsharp
type PlayRecord = {
    GameId: int             // Unique game identifier
    Step: int               // Step number within game
    CurrentScore: int       // Score before action
    CardsDrawn: int list    // All cards drawn so far
    Action: int             // 0=Stand, 1=Hit
    ResultScore: int        // Score after action
    Reward: float           // Game reward
    Done: bool              // Game ended flag
}
```

### Feature Engineering

The system extracts 12 features from each game state:
- **CurrentScore** (1 float) - Player's current score
- **CardHistogram[1-10]** (10 ints) - Count of each card value drawn
- **CardsDrawnCount** (1 int) - Total number of cards drawn

## Training Methods

### 1. Standard Rule-Based Trainer
- Analyzes gameplay data to learn statistical patterns
- Creates policy based on majority voting from training data
- Fallback to simple rule: Hit if score < 17, Stand otherwise
- **Performance**: ~99% accuracy

### 2. Neural Network Trainer
- Simple feedforward architecture: 12 → 32 → 32 → 2
- Adam optimizer with cross-entropy loss
- 100 epochs, batch size 32
- **Performance**: 100% accuracy

### 3. ResNet Trainer (Advanced)
- Residual network with skip connections: 12 → 64 → 4×ResBlocks → 2
- Advanced training features:
  - Learning rate scheduling (StepLR)
  - Early stopping with patience
  - Gradient clipping for stability
  - Weight decay regularization
- 150 epochs, batch size 64
- **Performance**: 100% accuracy

## Installation & Setup

### Prerequisites
- .NET 9.0 SDK
- F# compiler

### Dependencies
```xml
<PackageReference Include="FSharp.Data" Version="6.6.0" />
<PackageReference Include="TorchSharp" Version="0.105.1" />
<PackageReference Include="TorchSharp-cpu" Version="0.105.1" />
```

### Build
```bash
git clone <repository-url>
cd 21points
dotnet build
```

## Usage

### 1. Generate Training Data
```bash
# Run game simulator to generate gameplay data
./run-simulator.sh
# or
dotnet run --project GameSimulator/GameSimulator.fsproj
```

This creates `gameplay_log.csv` with gameplay records from 1000+ simulated games.

### 2. Train AI Models
```bash
# Run action predictor with interactive training method selection
./run-predictor.sh
# or
dotnet run --project ActionPredictor/ActionPredictor.fsproj
```

**Training Options:**
1. **Standard rule-based trainer** - Statistical learning approach
2. **Neural network trainer** - Simple feedforward network
3. **ResNet trainer** - Advanced residual network with skip connections

### 3. Test Trained Models
```bash
# Test a specific model
dotnet run --project ActionPredictor/ActionPredictor.fsproj -- --test model.pt test_data.csv

# List and test available models
dotnet run --project ActionPredictor/ActionPredictor.fsproj -- --test
```

## Model Performance Comparison

| Model Type | Architecture | Parameters | Accuracy | Training Time | Features |
|------------|-------------|------------|----------|---------------|----------|
| Rule-based | Statistical | N/A | ~99% | Fast | Interpretable |
| Neural Network | 12→32→32→2 | ~1,058 | 100% | Medium | Standard training |
| ResNet | 12→64→4×ResBlocks→2 | 2,978 | 100% | Slower | Advanced features |

### Advanced Training Features (ResNet)
- **Learning Rate Scheduling**: Reduces LR by 50% every 30 epochs
- **Early Stopping**: Stops training when validation doesn't improve for 20 epochs
- **Gradient Clipping**: Prevents exploding gradients (max norm = 1.0)
- **Weight Decay**: L2 regularization (1e-4) to prevent overfitting
- **Dropout**: 30% dropout in classification head

## Sample Output

### Training Progress
```
=== Training ResNet Model ===
ResNet Architecture: Input(12) -> ResBlocks(4x64) -> Classifier -> Output(2)
Training for 150 epochs with batch size 64

Epoch  30/150 - Train Loss: 0.0456 - Train Acc: 98.52% - Val Loss: 0.0359 - Val Acc: 99.66%
Epoch  55/150 - Train Loss: 0.0250 - Train Acc: 99.63% - Val Loss: 0.0205 - Val Acc: 100.00%
Early stopping at epoch 75 (best: 100.00% at epoch 55)

=== Final Metrics ===
Accuracy: 100.00%
Precision (Hit): 100.00%
Recall (Hit): 100.00%
F1-Score: 100.00%
```

### Sample Predictions
```
Score | Cards | Actual | Predicted | Confidence | Correct
------+-------+--------+-----------+------------+--------
   13 | 9,3,1   | Hit   | Hit   |    100.0% | ✓
   21 | 9,3,1,8 | Stand | Stand |     99.9% | ✓
   17 | 5,6,1,5 | Stand | Stand |     80.6% | ✓
```

## File Formats

### gameplay_log.csv
```csv
GameId,Step,CurrentScore,CardsDrawn,Action,ResultScore,Reward,Done
1,0,0,"",1,7,0.0,false
1,1,7,"7",1,15,0.0,false
1,2,15,"7;8",0,15,15.0,true
```

### Model Files
- **Neural Network Models**: `*.pt` files (TorchSharp format)
- **Rule-based Models**: `*.json` files with policy decisions

## Development Scripts

### Bash Scripts (Linux/macOS)
- `run-simulator.sh` - Run game simulator
- `run-predictor.sh` - Run action predictor
- `run-all.sh` - Run full pipeline
- `test-model.sh` - Test trained models

### PowerShell Scripts (Windows)
- `run-simulator.ps1` - Run game simulator  
- `run-predictor.ps1` - Run action predictor
- `run-all.ps1` - Run full pipeline
- `test-model.ps1` - Test trained models

## Key Algorithms

### Residual Block Implementation
```fsharp
override _.forward(x) =
    let residual = shortcut.forward(x)
    let out = x |> conv1.call |> relu |> conv2.call
    let result = out.add(residual)  // Skip connection
    relu(result)
```

### Feature Extraction
```fsharp
let extractFeatures (record: PlayRecord) =
    let histogram = Array.create 10 0
    for card in record.CardsDrawn do
        if card >= 1 && card <= 10 then
            histogram.[card - 1] <- histogram.[card - 1] + 1
    
    Array.concat [
        [| float record.CurrentScore |]
        Array.map float histogram
        [| float record.CardsDrawn.Length |]
    ]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Future Enhancements

- [ ] Multi-deck support
- [ ] Variable card distributions
- [ ] Reinforcement learning (Q-learning, Actor-Critic)
- [ ] GUI interface for interactive play
- [ ] Tournament mode with multiple AI strategies
- [ ] Real-time model comparison dashboard

## License

This project is open source. See LICENSE file for details.

## Technical Notes

- **Framework**: .NET 9.0 with F#
- **ML Library**: TorchSharp (PyTorch bindings for .NET)
- **Data Processing**: FSharp.Data for CSV handling
- **Architecture**: Functional programming with object-oriented ML components

## Citation

If you use this project in your research, please cite:

```
@software{21point_ai,
  title={21-Point Card Game AI with Neural Networks and ResNet},
  author={Your Name},
  year={2024},
  url={https://github.com/username/21points}
}
```