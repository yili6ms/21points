module ActionPredictor.ResNetTrainer

open System
open TorchSharp
open Types
open FeatureExtractor
open ResNet
open ModelPersistence

let trainResNet (records: PlayRecord list) =
    printfn "\n=== Training ResNet Model ==="
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
    
    // Create ResNet model with 4 residual blocks
    let model = createGameResNet(4)
    printfn "ResNet Architecture: Input(12) -> ResBlocks(4x64) -> Classifier -> Output(2)"
    
    // Setup training with different hyperparameters for ResNet
    let optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    let criterion = torch.nn.CrossEntropyLoss()
    let scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    let epochs = 150
    let batchSize = 64
    let numBatches = (trainingData.Length + batchSize - 1) / batchSize
    
    printfn "Training for %d epochs with batch size %d" epochs batchSize
    printfn "Number of batches per epoch: %d" numBatches
    printfn "Learning rate: 5e-4 with weight decay: 1e-4"
    printfn "LR scheduler: StepLR (step=30, gamma=0.5)"
    
    let mutable bestValAcc = 0.0
    let mutable bestEpoch = 0
    let mutable patienceCounter = 0
    let mutable shouldStop = false
    let patience = 20
    
    // Training loop with early stopping
    let mutable epoch = 1
    while epoch <= epochs && not shouldStop do
        model.train() |> ignore
        let mutable totalLoss = 0.0
        let mutable trainCorrect = 0
        let mutable trainTotal = 0
        
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
                
                // Calculate training accuracy for this batch
                let predictions = outputs.argmax(1)
                let correct = predictions.eq(batchLabels).sum().ToInt32()
                trainCorrect <- trainCorrect + correct
                trainTotal <- trainTotal + batchSize'
                
                loss.backward()
                
                // Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) |> ignore
                
                optimizer.step() |> ignore
                
                totalLoss <- totalLoss + loss.ToDouble()
        
        // Update learning rate
        scheduler.step()
        
        let avgLoss = totalLoss / float numBatches
        let trainAcc = float trainCorrect / float trainTotal * 100.0
        
        // Validation every 5 epochs or at the end
        if epoch % 5 = 0 || epoch = epochs then
            model.eval() |> ignore
            use _noGrad = torch.no_grad()
            
            let valOutputs = model.forward(valFeatures)
            let valLoss = criterion.call(valOutputs, valLabels).ToDouble()
            
            let predictions = valOutputs.argmax(1)
            let correct = predictions.eq(valLabels).sum().ToInt32()
            let valAcc = float correct / float validationData.Length * 100.0
            
            let currentLr = (optimizer.ParamGroups |> Seq.head).LearningRate
            
            printfn "Epoch %3d/%d - Train Loss: %.4f - Train Acc: %.2f%% - Val Loss: %.4f - Val Acc: %.2f%% - LR: %.6f" 
                epoch epochs avgLoss trainAcc valLoss valAcc currentLr
            
            // Early stopping logic
            if valAcc > bestValAcc then
                bestValAcc <- valAcc
                bestEpoch <- epoch
                patienceCounter <- 0
                // Save best model
                let bestModelPath = "best_resnet_model.pt"
                model.save(bestModelPath) |> ignore
            else
                patienceCounter <- patienceCounter + 1
                
            if patienceCounter >= patience then
                printfn "Early stopping at epoch %d (best: %.2f%% at epoch %d)" epoch bestValAcc bestEpoch
                shouldStop <- true
        else
            // Print training progress every 10 epochs
            if epoch % 10 = 0 then
                let currentLr = (optimizer.ParamGroups |> Seq.head).LearningRate
                printfn "Epoch %3d/%d - Train Loss: %.4f - Train Acc: %.2f%% - LR: %.6f" 
                    epoch epochs avgLoss trainAcc currentLr
        
        epoch <- epoch + 1
    
    // Load best model for final evaluation
    if System.IO.File.Exists("best_resnet_model.pt") then
        model.load("best_resnet_model.pt") |> ignore
        printfn "\nLoaded best model from epoch %d (Val Acc: %.2f%%)" bestEpoch bestValAcc
    
    // Final evaluation
    printfn "\n=== Final ResNet Evaluation ==="
    model.eval() |> ignore
    use _noGrad = torch.no_grad()
    
    let valOutputs = model.forward(valFeatures)
    let predictions = valOutputs.argmax(1)
    let probabilities = torch.nn.functional.softmax(valOutputs, 1)
    
    let correct = predictions.eq(valLabels).sum().ToInt32()
    let accuracy = float correct / float validationData.Length * 100.0
    
    printfn "Final Validation Accuracy: %.2f%%" accuracy
    
    // Detailed confusion matrix
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
    
    // Save the final model
    let modelName = sprintf "resnet_model_%s" (DateTime.Now.ToString("yyyyMMdd_HHmmss"))
    let modelPath = sprintf "%s.pt" modelName
    model.save(modelPath) |> ignore
    printfn "\nFinal model saved to: %s" modelPath
    
    // Show sample predictions with confidence scores
    printfn "\n=== Sample ResNet Predictions ==="
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
    
    // Create a prediction function using the trained ResNet
    let predictAction (features: float array) =
        model.eval() |> ignore
        use _noGrad = torch.no_grad()
        let inputTensor = torch.tensor(features).unsqueeze(0)
        let output = model.forward(inputTensor)
        let probabilities = torch.nn.functional.softmax(output, 1)
        let prediction = output.argmax(1).ToInt32()
        let confidence = probabilities.[0, prediction].ToSingle()
        (prediction, confidence)
    
    // Model complexity info
    let totalParams = 
        model.parameters() 
        |> Seq.sumBy (fun p -> p.numel())
    
    printfn "\n=== Model Information ==="
    printfn "Total parameters: %s" (totalParams.ToString("N0"))
    printfn "Best epoch: %d" bestEpoch
    printfn "Best validation accuracy: %.2f%%" bestValAcc
    
    (model, predictAction, accuracy)