module GameSimulator.Player

open Types

let simplePolicy (score: int) =
    if score < 17 then Hit else Stand

let makeDecision (state: GameState) =
    simplePolicy state.Score