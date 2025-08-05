module ActionPredictor.Types

type PlayRecord = {
    GameId: int
    Step: int
    CurrentScore: int
    CardsDrawn: int list
    Action: int
    ResultScore: int
    Reward: float
    Done: bool
}

type TrainingData = {
    Features: float32[]
    Label: int
}

type EvaluationResult = {
    Actual: int
    Predicted: int
    Correct: bool
}

type ConfusionMatrix = {
    TruePositive: int
    TrueNegative: int
    FalsePositive: int
    FalseNegative: int
}

type ModelMetrics = {
    Accuracy: float
    Precision: float
    Recall: float
    F1Score: float
    ConfusionMatrix: ConfusionMatrix
}

type BatchResult = {
    Loss: float
    Correct: int
    Total: int
}

type TrainingState = {
    Epoch: int
    BestValAcc: float
    BestEpoch: int
    PatienceCounter: int
    ShouldStop: bool
}