module GameSimulator.Game

open Types
open CardDeck
open Player

let initGame() = {
    Score = 0
    CardsDrawn = []
    IsOver = false
}

let executeAction (state: GameState) (action: Action) =
    match action with
    | Stand -> 
        { state with IsOver = true }
    | Hit ->
        let card = takeCard()
        let newScore = state.Score + card
        let newCardsDrawn = state.CardsDrawn @ [card]
        
        if newScore > 21 then
            { Score = newScore; CardsDrawn = newCardsDrawn; IsOver = true }
        else
            { Score = newScore; CardsDrawn = newCardsDrawn; IsOver = false }

let calculateReward (score: int) =
    if score > 21 then 0.0
    else float score

let playGame (gameId: int) =
    let mutable state = initGame()
    let mutable records = []
    let mutable step = 0
    
    while not state.IsOver do
        let action = makeDecision state
        let actionInt = action.ToInt()
        
        let beforeScore = state.Score
        state <- executeAction state action
        let reward = if state.IsOver then calculateReward state.Score else 0.0
        
        let record = {
            GameId = gameId
            Step = step
            CurrentScore = beforeScore
            CardsDrawn = List.take (List.length state.CardsDrawn - (if action = Hit then 1 else 0)) state.CardsDrawn
            Action = actionInt
            ResultScore = state.Score
            Reward = reward
            Done = state.IsOver
        }
        
        records <- records @ [record]
        step <- step + 1
    
    records