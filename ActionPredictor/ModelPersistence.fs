module ActionPredictor.ModelPersistence

open System.IO
open System.Text.Json
open Types

type PolicyModel = {
    Name: string
    TrainingDate: string
    TrainingDataSize: int
    ValidationAccuracy: float
    HitDecisions: int list // List of scores where model decides to Hit
    StandDecisions: int list // List of scores where model decides to Stand
    ScoreStats: Map<int, int * int> // Map of score -> (hitCount, standCount)
}

let saveModel (modelName: string) (hitDecisions: int list) (standDecisions: int list) (scoreStats: Map<int, int * int>) (trainingSize: int) (accuracy: float) =
    let model = {
        Name = modelName
        TrainingDate = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
        TrainingDataSize = trainingSize
        ValidationAccuracy = accuracy
        HitDecisions = hitDecisions
        StandDecisions = standDecisions
        ScoreStats = scoreStats
    }
    
    let fileName = sprintf "%s_model.json" modelName
    let options = JsonSerializerOptions()
    options.WriteIndented <- true
    
    let json = JsonSerializer.Serialize(model, options)
    File.WriteAllText(fileName, json)
    
    printfn "\n=== Model Saved ==="
    printfn "File: %s" fileName
    printfn "Training Data Size: %d" trainingSize
    printfn "Validation Accuracy: %.2f%%" accuracy
    printfn "Hit Decisions: %s" (hitDecisions |> List.map string |> String.concat ", ")
    printfn "Stand Decisions: %s" (standDecisions |> List.map string |> String.concat ", ")
    
    fileName

let loadModel (fileName: string) =
    if not (File.Exists(fileName)) then
        printfn "Error: Model file %s not found" fileName
        None
    else
        try
            let json = File.ReadAllText(fileName)
            let model = JsonSerializer.Deserialize<PolicyModel>(json)
            
            printfn "\n=== Model Loaded ==="
            printfn "Name: %s" model.Name
            printfn "Training Date: %s" model.TrainingDate
            printfn "Training Data Size: %d" model.TrainingDataSize
            printfn "Original Validation Accuracy: %.2f%%" model.ValidationAccuracy
            
            // Create policy function from loaded model
            let policy score =
                if List.contains score model.HitDecisions then 1
                elif List.contains score model.StandDecisions then 0
                else if score < 17 then 1 else 0 // Fallback rule
            
            Some (model, policy)
        with
        | ex ->
            printfn "Error loading model: %s" ex.Message
            None

let listModels () =
    let modelFiles = Directory.GetFiles(".", "*_model.json")
    
    if modelFiles.Length = 0 then
        printfn "No saved models found"
    else
        printfn "\n=== Available Models ==="
        for file in modelFiles do
            try
                let json = File.ReadAllText(file)
                let model = JsonSerializer.Deserialize<PolicyModel>(json)
                printfn "%s - %s (Accuracy: %.2f%%, Size: %d)" 
                    (Path.GetFileName(file))
                    model.TrainingDate
                    model.ValidationAccuracy
                    model.TrainingDataSize
            with
            | _ -> printfn "%s - (corrupted)" (Path.GetFileName(file))