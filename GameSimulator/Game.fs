module GameSimulator.Game

open Types
open CardDeck
open Player

// Pure game state initialization
let initGame = {
    Score = 0
    CardsDrawn = []
    IsOver = false
}

// Pure action execution with explicit random state threading
let executeActionPure (state: GameState) (action: Action) (randomState: RandomState) =
    match action with
    | Stand -> 
        ({ state with IsOver = true }, randomState)
    | Hit ->
        let (card, newRandomState) = takeCardPure randomState
        let newScore = state.Score + card
        let newCardsDrawn = state.CardsDrawn @ [card]
        
        let newState = 
            if newScore > 21 then
                { Score = newScore; CardsDrawn = newCardsDrawn; IsOver = true }
            else
                { Score = newScore; CardsDrawn = newCardsDrawn; IsOver = false }
        
        (newState, newRandomState)

// Impure wrapper for backward compatibility
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

// Pure reward calculation
let calculateReward (score: int) =
    if score > 21 then 0.0
    else float score

// Pure functional game play with explicit state threading
let rec playGamePure (gameId: int) (state: GameState) (randomState: RandomState) (step: int) (records: PlayRecord list) =
    if state.IsOver then
        records
    else
        let action = makeDecision state
        let actionInt = action.ToInt()
        let beforeScore = state.Score
        
        let (newState, newRandomState) = executeActionPure state action randomState
        let reward = if newState.IsOver then calculateReward newState.Score else 0.0
        
        let record = {
            GameId = gameId
            Step = step
            CurrentScore = beforeScore
            CardsDrawn = state.CardsDrawn
            Action = actionInt
            ResultScore = newState.Score
            Reward = reward
            Done = newState.IsOver
        }
        
        let newRecords = records @ [record]
        playGamePure gameId newState newRandomState (step + 1) newRecords

// Pure game play with seed
let playGameWithSeed (gameId: int) (seed: int) =
    let randomState = initRandomState seed
    playGamePure gameId initGame randomState 0 []

// Impure wrapper for backward compatibility
let playGame (gameId: int) =
    let seed = int (System.DateTime.Now.Ticks % int64 System.Int32.MaxValue)
    playGameWithSeed gameId seed