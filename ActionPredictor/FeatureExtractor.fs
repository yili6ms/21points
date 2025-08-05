module ActionPredictor.FeatureExtractor

open Types

let extractFeatures (record: PlayRecord) : TrainingData =
    let cardHistogram = Array.create 10 0
    
    for card in record.CardsDrawn do
        if card >= 1 && card <= 10 then
            cardHistogram.[card - 1] <- cardHistogram.[card - 1] + 1
    
    let features = 
        [|
            float32 record.CurrentScore
            yield! cardHistogram |> Array.map float32
            float32 record.CardsDrawn.Length
        |]
    
    { Features = features; Label = record.Action }

let convertToTensors (records: PlayRecord list) =
    records |> List.map extractFeatures