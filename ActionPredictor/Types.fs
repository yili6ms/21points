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