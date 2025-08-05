module ActionPredictor.StandardTrainer

open System
open Types
open FeatureExtractor
open ModelPersistence

let trainModel (records: PlayRecord list) =
    printfn "\n=== Training Standard Model ==="
    
    let data = convertToTensors records
    
    // Split data 80/20
    let splitRatio = 0.8
    let splitIndex = int (float data.Length * splitRatio)
    let trainingData = data |> List.take splitIndex
    let validationData = data |> List.skip splitIndex
    
    printfn "Training samples: %d" trainingData.Length
    printfn "Validation samples: %d" validationData.Length
    
    // Analyze training data to build a more sophisticated policy
    let hitStats = 
        trainingData 
        |> List.filter (fun d -> d.Label = 1)
        |> List.groupBy (fun d -> int d.Features.[0]) // Group by current score
        |> List.map (fun (score, items) -> (score, items.Length))
        |> Map.ofList
    
    let standStats = 
        trainingData 
        |> List.filter (fun d -> d.Label = 0)
        |> List.groupBy (fun d -> int d.Features.[0])
        |> List.map (fun (score, items) -> (score, items.Length))
        |> Map.ofList
    
    // Create learned policy based on data patterns
    let learnedPolicy score =
        let hitCount = hitStats |> Map.tryFind score |> Option.defaultValue 0
        let standCount = standStats |> Map.tryFind score |> Option.defaultValue 0
        let total = hitCount + standCount
        
        if total = 0 then
            // Fallback to simple rule
            if score < 17 then 1 else 0
        else
            // Use majority vote from training data
            if hitCount > standCount then 1 else 0
    
    // Evaluate on validation data
    let mutable correct = 0
    let mutable total = 0
    
    for item in validationData do
        let score = int item.Features.[0]
        let prediction = learnedPolicy score
        if prediction = item.Label then
            correct <- correct + 1
        total <- total + 1
    
    let accuracy = float correct / float total * 100.0
    printfn "Learned policy accuracy: %.2f%%" accuracy
    
    // Show policy decisions for different scores
    printfn "\n=== Policy Analysis ==="
    printfn "Score | Decision | Hit Count | Stand Count"
    printfn "------+----------+-----------+------------"
    for score in 1..21 do
        let hitCount = hitStats |> Map.tryFind score |> Option.defaultValue 0
        let standCount = standStats |> Map.tryFind score |> Option.defaultValue 0
        let decision = if learnedPolicy score = 1 then "Hit  " else "Stand"
        printfn "%5d | %s   | %9d | %10d" score decision hitCount standCount
    
    // Show confusion matrix
    let mutable truePositive = 0
    let mutable trueNegative = 0
    let mutable falsePositive = 0
    let mutable falseNegative = 0
    
    for item in validationData do
        let score = int item.Features.[0]
        let prediction = learnedPolicy score
        let actual = item.Label
        
        match prediction, actual with
        | 0, 0 -> trueNegative <- trueNegative + 1
        | 1, 1 -> truePositive <- truePositive + 1
        | 0, 1 -> falseNegative <- falseNegative + 1
        | 1, 0 -> falsePositive <- falsePositive + 1
        | _ -> ()
    
    printfn "\n=== Confusion Matrix ==="
    printfn "                Predicted"
    printfn "             Stand    Hit"
    printfn "Actual Stand  %4d   %4d" trueNegative falsePositive
    printfn "Actual Hit    %4d   %4d" falseNegative truePositive
    
    let precision = 
        if truePositive + falsePositive > 0 then
            float truePositive / float (truePositive + falsePositive) * 100.0
        else 0.0
    let recall = 
        if truePositive + falseNegative > 0 then
            float truePositive / float (truePositive + falseNegative) * 100.0
        else 0.0
    
    printfn "\n=== Final Metrics ==="
    printfn "Accuracy: %.2f%%" accuracy
    printfn "Precision (Hit): %.2f%%" precision
    printfn "Recall (Hit): %.2f%%" recall
    
    // Prepare data for model saving
    let hitDecisions = [1..21] |> List.filter (fun score -> learnedPolicy score = 1)
    let standDecisions = [1..21] |> List.filter (fun score -> learnedPolicy score = 0)
    let scoreStatsMap = 
        [1..21] 
        |> List.map (fun score -> 
            let hitCount = hitStats |> Map.tryFind score |> Option.defaultValue 0
            let standCount = standStats |> Map.tryFind score |> Option.defaultValue 0
            (score, (hitCount, standCount)))
        |> Map.ofList
    
    // Save the model
    let modelName = sprintf "policy_%s" (DateTime.Now.ToString("yyyyMMdd_HHmmss"))
    let savedFile = saveModel modelName hitDecisions standDecisions scoreStatsMap trainingData.Length accuracy
    
    // Show sample predictions
    printfn "\n=== Sample Predictions ==="
    printfn "Score | Cards | Actual | Predicted | Correct"
    printfn "------+-------+--------+-----------+--------"
    
    let samples = 
        List.zip (records |> List.skip splitIndex) validationData 
        |> List.take (min 20 (List.length validationData))
    
    for record, item in samples do
        let score = int item.Features.[0]
        let prediction = learnedPolicy score
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
    
    learnedPolicy