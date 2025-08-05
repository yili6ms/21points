module ActionPredictor.ResNet

open TorchSharp
open type torch.nn

// Residual Block for the ResNet
type ResidualBlock(inChannels: int64, outChannels: int64) as this =
    inherit Module<torch.Tensor, torch.Tensor>("ResidualBlock")
    
    let conv1 = Linear(inChannels, outChannels)
    let conv2 = Linear(outChannels, outChannels)
    
    // Skip connection - identity or projection
    let shortcut = 
        if inChannels <> outChannels then
            Linear(inChannels, outChannels) :> Module<torch.Tensor, torch.Tensor>
        else
            Identity() :> Module<torch.Tensor, torch.Tensor>
    
    do
        this.RegisterComponents()
    
    override _.forward(x) =
        let residual = shortcut.forward(x)
        
        let out = x
                  |> conv1.call
                  |> torch.nn.functional.relu
                  |> conv2.call
        
        // Add residual connection
        let result = out.add(residual)
        torch.nn.functional.relu(result)

// Simple ResNet Model for 21-point game
type ResNetModel(inputSize: int64, numClasses: int64, numBlocks: int) as this =
    inherit Module<torch.Tensor, torch.Tensor>("ResNetModel")
    
    let hiddenSize = 64L
    
    // Initial layer to expand input to hidden size
    let inputLayer = Linear(inputSize, hiddenSize)
    
    // Create residual blocks
    let resBlocks = 
        [for i in 0..numBlocks-1 ->
            ResidualBlock(hiddenSize, hiddenSize)]
    
    // Final classification layers
    let fc1 = Linear(hiddenSize, hiddenSize / 2L)
    let dropout = Dropout(0.3)
    let fc2 = Linear(hiddenSize / 2L, numClasses)
    
    do
        this.RegisterComponents()
    
    override _.forward(x) =
        // Initial processing
        let mutable out = x
                          |> inputLayer.call
                          |> torch.nn.functional.relu
        
        // Pass through residual blocks
        for block in resBlocks do
            out <- block.forward(out)
        
        // Final classification
        out |> fc1.call
            |> torch.nn.functional.relu
            |> dropout.call
            |> fc2.call

// Factory function to create ResNet model
let createResNetModel(inputSize: int64, numClasses: int64, numBlocks: int) =
    new ResNetModel(inputSize, numClasses, numBlocks)

// Specific factory for 21-point game (12 features -> 2 classes)
let createGameResNet(numBlocks: int) =
    createResNetModel(12L, 2L, numBlocks)