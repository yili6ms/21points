module GameSimulator.Types

type Action = 
    | Stand 
    | Hit
    
    member this.ToInt() = 
        match this with
        | Stand -> 0
        | Hit -> 1
    
    static member FromInt(value: int) =
        match value with
        | 0 -> Stand
        | _ -> Hit

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

type GameState = {
    Score: int
    CardsDrawn: int list
    IsOver: bool
}