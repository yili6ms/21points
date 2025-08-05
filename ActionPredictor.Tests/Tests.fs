module ActionPredictor.Tests

open Xunit
open ActionPredictor.Types
open ActionPredictor.FeatureExtractor
open ActionPredictor.DataLoader

[<Fact>]
let ``extractFeatures should create correct feature vector size`` () =
    let record = {
        GameId = 1
        Step = 0
        CurrentScore = 15
        CardsDrawn = [7; 8]
        Action = 1
        ResultScore = 15
        Reward = 0.0
        Done = false
    }
    
    let trainingData = extractFeatures record
    
    // Features should be: CurrentScore(1) + CardHistogram(10) + CardsDrawnCount(1) = 12
    Assert.Equal(12, trainingData.Features.Length)
    Assert.Equal(1, trainingData.Label)

[<Fact>]
let ``extractFeatures should correctly encode current score`` () =
    let record = {
        GameId = 1
        Step = 0
        CurrentScore = 15
        CardsDrawn = []
        Action = 0
        ResultScore = 15
        Reward = 0.0
        Done = false
    }
    
    let trainingData = extractFeatures record
    Assert.Equal(15.0f, trainingData.Features.[0])

[<Fact>]
let ``extractFeatures should correctly build card histogram`` () =
    let record = {
        GameId = 1
        Step = 0
        CurrentScore = 15
        CardsDrawn = [7; 8; 7] // Two 7s and one 8
        Action = 1
        ResultScore = 15
        Reward = 0.0
        Done = false
    }
    
    let trainingData = extractFeatures record
    
    // Card histogram starts at index 1: [1-count, 2-count, ..., 10-count]
    Assert.Equal(0.0f, trainingData.Features.[1]) // No 1s
    Assert.Equal(2.0f, trainingData.Features.[7]) // Two 7s (index 7 = card 7)
    Assert.Equal(1.0f, trainingData.Features.[8]) // One 8 (index 8 = card 8)

[<Fact>]
let ``extractFeatures should correctly encode cards drawn count`` () =
    let record = {
        GameId = 1
        Step = 0
        CurrentScore = 15
        CardsDrawn = [7; 8; 3]
        Action = 1
        ResultScore = 15
        Reward = 0.0
        Done = false
    }
    
    let trainingData = extractFeatures record
    
    // Cards drawn count should be at index 11 (last element)
    Assert.Equal(3.0f, trainingData.Features.[11])

[<Fact>]
let ``parseCardsDrawn should handle empty string`` () =
    let result = parseCardsDrawn ""
    Assert.Empty(result)

[<Fact>]
let ``parseCardsDrawn should handle single card`` () =
    let result = parseCardsDrawn "7"
    Assert.Single(result)
    result.[0] |> ignore
    Assert.Equal(7, result.[0])

[<Fact>]
let ``parseCardsDrawn should handle multiple cards`` () =
    let result = parseCardsDrawn "7;8;10"
    Assert.Equal(3, result.Length)
    Assert.Equal<int list>([7; 8; 10], result)

[<Fact>]
let ``parseCardsDrawn should handle whitespace`` () =
    let result = parseCardsDrawn "  "
    Assert.Empty(result)

[<Fact>]
let ``parseCardsDrawn should filter empty segments`` () =
    let result = parseCardsDrawn "7;;8;"
    Assert.Equal(2, result.Length)
    Assert.Equal<int list>([7; 8], result)

[<Fact>]
let ``convertToTensors should process multiple records`` () =
    let records = [
        {
            GameId = 1; Step = 0; CurrentScore = 0; CardsDrawn = []
            Action = 1; ResultScore = 7; Reward = 0.0; Done = false
        }
        {
            GameId = 1; Step = 1; CurrentScore = 7; CardsDrawn = [7]
            Action = 0; ResultScore = 7; Reward = 7.0; Done = true
        }
    ]
    
    let trainingData = convertToTensors records
    
    Assert.Equal(2, trainingData.Length)
    Assert.Equal(1, trainingData.[0].Label)
    Assert.Equal(0, trainingData.[1].Label)

[<Fact>]
let ``TrainingData should store features and labels correctly`` () =
    let features = [| 1.0f; 2.0f; 3.0f |]
    let label = 1
    
    let trainingData = { Features = features; Label = label }
    
    Assert.Equal<float32[]>(features, trainingData.Features)
    Assert.Equal(label, trainingData.Label)

[<Fact>]
let ``EvaluationResult should correctly identify correct predictions`` () =
    let correctResult = { Actual = 1; Predicted = 1; Correct = true }
    let incorrectResult = { Actual = 1; Predicted = 0; Correct = false }
    
    Assert.True(correctResult.Correct)
    Assert.False(incorrectResult.Correct)

[<Fact>]
let ``ConfusionMatrix should have correct structure`` () =
    let matrix = { TruePositive = 10; TrueNegative = 15; FalsePositive = 3; FalseNegative = 2 }
    
    Assert.Equal(10, matrix.TruePositive)
    Assert.Equal(15, matrix.TrueNegative)
    Assert.Equal(3, matrix.FalsePositive)
    Assert.Equal(2, matrix.FalseNegative)
