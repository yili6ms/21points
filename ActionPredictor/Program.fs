open System
open ActionPredictor.Types
open ActionPredictor.DataLoader
open ActionPredictor.StandardTrainer
open ActionPredictor.NeuralNetworkTrainer
open ActionPredictor.ResNetTrainer
open ActionPredictor.ModelTester

[<EntryPoint>]
let main argv =
    printfn "=== 21-Point Action Predictor ==="
    
    if argv.Length > 0 && argv.[0] = "--test" then
        // Test mode
        if argv.Length >= 3 then
            let modelFile = argv.[1]
            let testFile = argv.[2]
            testModel modelFile testFile
        else
            listAndTestModel()
    else
        // Training mode
        printfn "Loading gameplay data for training..."
        
        let filename = 
            if argv.Length > 0 then argv.[0] 
            else "gameplay_log.csv"
        
        let records = loadFromCsv filename
        
        if records.IsEmpty then
            printfn "Error: No data loaded from %s" filename
            1
        else
            printStats records
            
            printfn "\nSelect training method:"
            printfn "1. Standard rule-based trainer"
            printfn "2. Neural network trainer"
            printfn "3. ResNet trainer"
            printf "Enter choice (1, 2, or 3): "
            let choice = Console.ReadLine()
            
            match choice with
            | "2" ->
                printfn "\nUsing Neural Network Trainer..."
                let (model, predictFn, accuracy) = trainNeuralNetwork records
                printfn "\n=== Neural Network Training Complete ==="
                printfn "Final accuracy: %.2f%%" accuracy
            | "3" ->
                printfn "\nUsing ResNet Trainer..."
                let (model, predictFn, accuracy) = trainResNet records
                printfn "\n=== ResNet Training Complete ==="
                printfn "Final accuracy: %.2f%%" accuracy
            | _ ->
                printfn "\nUsing Standard Rule-based Trainer..."
                let learnedPolicy = trainModel records
                printfn "\n=== Standard Training Complete ==="
            
            0