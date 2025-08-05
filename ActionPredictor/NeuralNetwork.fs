module ActionPredictor.NeuralNetwork

open TorchSharp
open type torch
open type torch.nn
open type torch.nn.functional

type ActionPredictorModel() as this =
    inherit Module<torch.Tensor, torch.Tensor>("ActionPredictor")
    
    let fc1 = Linear(12L, 32L)
    let fc2 = Linear(32L, 32L)
    let fc3 = Linear(32L, 2L)
    
    do
        this.RegisterComponents()
        
    override _.forward(x) =
        x
        |> fc1.call
        |> relu
        |> fc2.call
        |> relu
        |> fc3.call

let createModel() =
    new ActionPredictorModel()