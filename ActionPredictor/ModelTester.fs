module ActionPredictor.ModelTester

open System
open Types
open FeatureExtractor
open ModelPersistence
open DataLoader

let testModel (modelFile: string) (testDataFile: string) =
    printfn "=== Model Testing ==="
    printfn "Model: %s" modelFile
    printfn "Test Data: %s" testDataFile
    
    // Load model
    match loadModel modelFile with
    | None -> 
        printfn "Failed to load model"
        1
    | Some (modelInfo, policy) ->
        // Load test data
        let testRecords = loadFromCsv testDataFile
        
        if testRecords.IsEmpty then
            printfn "Error: No test data loaded"
            1
        else
            printfn "\n=== Test Data Statistics ==="
            printStats testRecords
            
            // Convert to features
            let testData = convertToTensors testRecords
            
            // Test the model
            let mutable correct = 0
            let mutable total = 0
            let mutable truePositive = 0
            let mutable trueNegative = 0
            let mutable falsePositive = 0
            let mutable falseNegative = 0
            
            for item in testData do
                let score = int item.Features.[0]
                let prediction = policy score
                let actual = item.Label
                
                if prediction = actual then
                    correct <- correct + 1
                
                match prediction, actual with
                | 0, 0 -> trueNegative <- trueNegative + 1
                | 1, 1 -> truePositive <- truePositive + 1
                | 0, 1 -> falseNegative <- falseNegative + 1
                | 1, 0 -> falsePositive <- falsePositive + 1
                | _ -> ()
                
                total <- total + 1
            
            // Calculate metrics
            let accuracy = float correct / float total * 100.0
            let precision = 
                if truePositive + falsePositive > 0 then
                    float truePositive / float (truePositive + falsePositive) * 100.0
                else 0.0
            let recall = 
                if truePositive + falseNegative > 0 then
                    float truePositive / float (truePositive + falseNegative) * 100.0
                else 0.0
            let f1Score = 
                if precision + recall > 0.0 then
                    2.0 * precision * recall / (precision + recall)
                else 0.0
            
            printfn "\n=== Test Results ==="
            printfn "Test samples: %d" total
            printfn "Correct predictions: %d" correct
            printfn "Accuracy: %.2f%%" accuracy
            printfn "Precision (Hit): %.2f%%" precision
            printfn "Recall (Hit): %.2f%%" recall
            printfn "F1-Score: %.2f%%" f1Score
            
            printfn "\n=== Confusion Matrix ==="
            printfn "                Predicted"
            printfn "             Stand    Hit"
            printfn "Actual Stand  %4d   %4d" trueNegative falsePositive
            printfn "Actual Hit    %4d   %4d" falseNegative truePositive
            
            // Show detailed breakdown by score
            printfn "\n=== Performance by Score ==="
            printfn "Score | Test Count | Correct | Accuracy"
            printfn "------+------------+---------+---------"
            
            let scoreGroups = testData |> List.groupBy (fun d -> int d.Features.[0])
            for score, items in scoreGroups |> List.sortBy fst do
                let scoreCorrect = 
                    items 
                    |> List.filter (fun item -> policy score = item.Label)
                    |> List.length
                let scoreTotal = items.Length
                let scoreAccuracy = float scoreCorrect / float scoreTotal * 100.0
                printfn "%5d | %10d | %7d | %6.1f%%" score scoreTotal scoreCorrect scoreAccuracy
            
            // Show sample predictions
            printfn "\n=== Sample Test Predictions ==="
            printfn "Score | Cards | Actual | Predicted | Correct"
            printfn "------+-------+--------+-----------+--------"
            
            let samples = 
                List.zip testRecords testData 
                |> List.take (min 20 (List.length testData))
            
            for record, item in samples do
                let score = int item.Features.[0]
                let prediction = policy score
                let actual = item.Label
                
                let actionToString a = if a = 0 then "Stand" else "Hit  "
                let cardsStr = 
                    record.CardsDrawn 
                    |> List.map string 
                    |> String.concat ","
                    |> fun s -> if s.Length > 7 then s.Substring(0, 5) + ".." else s.PadRight(7)
                
                let correct = if prediction = actual then "✓" else "✗"
                
                printfn "%5d | %s | %s | %s | %s" 
                    record.CurrentScore 
                    cardsStr
                    (actionToString actual)
                    (actionToString prediction)
                    correct
            
            // Compare with original training performance
            printfn "\n=== Performance Comparison ==="
            printfn "Training Accuracy: %.2f%%" modelInfo.ValidationAccuracy
            printfn "Test Accuracy:     %.2f%%" accuracy
            let accuracyDrop = modelInfo.ValidationAccuracy - accuracy
            printfn "Accuracy Drop:     %.2f%%" accuracyDrop
            
            if accuracyDrop > 5.0 then
                printfn "⚠️  Significant accuracy drop detected - model may be overfitting"
            elif accuracyDrop < 1.0 then
                printfn "✅ Model generalizes well to new data"
            else
                printfn "📊 Model shows reasonable generalization"
            
            0

let listAndTestModel () =
    listModels()
    
    printfn "\nEnter model filename to test (or press Enter to skip): "
    let modelFile = Console.ReadLine()
    
    if String.IsNullOrWhiteSpace(modelFile) then
        printfn "Skipping model test"
        0
    else
        printfn "Enter test data filename (default: gameplay_log.csv): "
        let testFile = Console.ReadLine()
        let testDataFile = if String.IsNullOrWhiteSpace(testFile) then "gameplay_log.csv" else testFile
        
        testModel modelFile testDataFile