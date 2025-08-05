module ActionPredictor.NeuralNetworkTrainer

open System
open TorchSharp
open Types
open FeatureExtractor
open NeuralNetwork
open ModelPersistence

let trainNeuralNetwork (records: PlayRecord list) =
    printfn "\n=== Training Neural Network Model ==="
    printfn "Using CPU device for training"
    
    let data = convertToTensors records
    
    // Split data 80/20
    let splitRatio = 0.8
    let splitIndex = int (float data.Length * splitRatio)
    let trainingData = data |> List.take splitIndex
    let validationData = data |> List.skip splitIndex
    
    printfn "Training samples: %d" trainingData.Length
    printfn "Validation samples: %d" validationData.Length
    
    // Convert to tensors
    let trainFeatures = 
        trainingData 
        |> List.map (fun d -> d.Features) 
        |> array2D
        |> torch.tensor
    
    let trainLabels = 
        trainingData 
        |> List.map (fun d -> int64 d.Label) 
        |> Array.ofList
        |> torch.tensor
    
    let valFeatures = 
        validationData 
        |> List.map (fun d -> d.Features) 
        |> array2D  
        |> torch.tensor
    
    let valLabels = 
        validationData 
        |> List.map (fun d -> int64 d.Label) 
        |> Array.ofList
        |> torch.tensor
    
    // Create model
    let model = createModel()
    
    // Setup training
    let optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    let criterion = torch.nn.CrossEntropyLoss()
    
    let epochs = 100
    let batchSize = 32
    let numBatches = (trainingData.Length + batchSize - 1) / batchSize
    
    printfn "Training for %d epochs with batch size %d" epochs batchSize
    printfn "Number of batches per epoch: %d" numBatches
    
    // Training loop
    for epoch in 1..epochs do
        model.train() |> ignore
        let mutable totalLoss = 0.0
        
        // Mini-batch training
        for batch in 0..(numBatches - 1) do
            let startIdx = batch * batchSize
            let endIdx = min (startIdx + batchSize) trainingData.Length
            let batchSize' = endIdx - startIdx
            
            if batchSize' > 0 then
                let batchFeatures = trainFeatures.slice(0, startIdx, endIdx, 1)
                let batchLabels = trainLabels.slice(0, startIdx, endIdx, 1)
                
                optimizer.zero_grad()
                
                let outputs = model.forward(batchFeatures)
                let loss = criterion.call(outputs, batchLabels)
                
                loss.backward()
                optimizer.step() |> ignore
                
                totalLoss <- totalLoss + loss.ToDouble()
        
        let avgLoss = totalLoss / float numBatches
        
        // Validation every 10 epochs
        if epoch % 10 = 0 || epoch = epochs then
            model.eval() |> ignore
            use _noGrad = torch.no_grad()
            
            let valOutputs = model.forward(valFeatures)
            let valLoss = criterion.call(valOutputs, valLabels).ToDouble()
            
            let predictions = valOutputs.argmax(1)
            let correct = predictions.eq(valLabels).sum().ToInt32()
            let accuracy = float correct / float validationData.Length * 100.0
            
            printfn "Epoch %3d/%d - Train Loss: %.4f - Val Loss: %.4f - Val Acc: %.2f%%" 
                epoch epochs avgLoss valLoss accuracy
    
    // Final evaluation
    printfn "\n=== Final Neural Network Evaluation ==="
    model.eval() |> ignore
    use _noGrad = torch.no_grad()
    
    let valOutputs = model.forward(valFeatures)
    let predictions = valOutputs.argmax(1)
    let probabilities = torch.nn.functional.softmax(valOutputs, 1)
    
    let correct = predictions.eq(valLabels).sum().ToInt32()
    let accuracy = float correct / float validationData.Length * 100.0
    
    printfn "Final Validation Accuracy: %.2f%%" accuracy
    
    // Confusion matrix
    let mutable truePositive = 0
    let mutable trueNegative = 0
    let mutable falsePositive = 0
    let mutable falseNegative = 0
    
    let predArray = predictions.data<int64>().ToArray()
    let actualArray = valLabels.data<int64>().ToArray()
    
    for i in 0..(predArray.Length - 1) do
        match predArray.[i], actualArray.[i] with
        | 0L, 0L -> trueNegative <- trueNegative + 1
        | 1L, 1L -> truePositive <- truePositive + 1
        | 0L, 1L -> falseNegative <- falseNegative + 1
        | 1L, 0L -> falsePositive <- falsePositive + 1
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
    let f1Score = 
        if precision + recall > 0.0 then
            2.0 * precision * recall / (precision + recall)
        else 0.0
    
    printfn "\n=== Final Metrics ==="
    printfn "Accuracy: %.2f%%" accuracy
    printfn "Precision (Hit): %.2f%%" precision
    printfn "Recall (Hit): %.2f%%" recall
    printfn "F1-Score: %.2f%%" f1Score
    
    // Save the model
    let modelName = sprintf "nn_model_%s" (DateTime.Now.ToString("yyyyMMdd_HHmmss"))
    let modelPath = sprintf "%s.pt" modelName
    model.save(modelPath) |> ignore
    printfn "\nModel saved to: %s" modelPath
    
    // Show sample predictions with confidence
    printfn "\n=== Sample Neural Network Predictions ==="
    printfn "Score | Cards | Actual | Predicted | Confidence | Correct"
    printfn "------+-------+--------+-----------+------------+--------"
    
    let probArray = probabilities.data<float32>().ToArray()
    let samples = 
        List.zip (records |> List.skip splitIndex) validationData 
        |> List.take (min 20 (List.length validationData))
    
    for i, (record, item) in samples |> List.indexed do
        let prediction = int predArray.[i]
        let actual = item.Label
        let confidence = probArray.[i * 2 + prediction] * 100.0f
        
        let actionToString a = if a = 0 then "Stand" else "Hit  "
        let cardsStr = 
            record.CardsDrawn 
            |> List.map string 
            |> String.concat ","
            |> fun s -> if s.Length > 7 then s.Substring(0, 5) + ".." else s.PadRight(7)
        
        let correct = if prediction = actual then "✓" else "✗"
        
        printfn "%5d | %s | %s | %s | %8.1f%% | %s" 
            record.CurrentScore 
            cardsStr
            (actionToString actual)
            (actionToString prediction)
            confidence
            correct
    
    // Create a prediction function using the trained model
    let predictAction (features: float array) =
        model.eval() |> ignore
        use _noGrad = torch.no_grad()
        let inputTensor = torch.tensor(features).unsqueeze(0)
        let output = model.forward(inputTensor)
        let probabilities = torch.nn.functional.softmax(output, 1)
        let prediction = output.argmax(1).ToInt32()
        let confidence = probabilities.[0, prediction].ToSingle()
        (prediction, confidence)
    
    (model, predictAction, accuracy)