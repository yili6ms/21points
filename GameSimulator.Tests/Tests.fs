module GameSimulator.Tests

open Xunit
open GameSimulator.Types
open GameSimulator.CardDeck
open GameSimulator.Game

[<Fact>]
let ``Action ToInt should return correct values`` () =
    Assert.Equal(0, Stand.ToInt())
    Assert.Equal(1, Hit.ToInt())

[<Fact>]
let ``Action FromInt should return correct actions`` () =
    Assert.Equal(Stand, Action.FromInt(0))
    Assert.Equal(Hit, Action.FromInt(1))
    Assert.Equal(Hit, Action.FromInt(999)) // Any non-zero value should be Hit

[<Fact>]
let ``initRandomState should create consistent state`` () =
    let state1 = initRandomState 42
    let state2 = initRandomState 42
    Assert.Equal(state1.Seed, state2.Seed)
    Assert.Equal(state1.Next, state2.Next)

[<Fact>]
let ``takeCardPure should generate cards 1-10`` () =
    let randomState = initRandomState 42
    let mutable currentState = randomState
    let mutable allCardsValid = true
    
    // Test 100 card draws
    for _ in 1..100 do
        let (card, newState) = takeCardPure currentState
        if card < 1 || card > 10 then
            allCardsValid <- false
        currentState <- newState
    
    Assert.True(allCardsValid)

[<Fact>]
let ``takeCardPure with same seed should be deterministic`` () =
    let state1 = initRandomState 42
    let state2 = initRandomState 42
    
    let (card1, _) = takeCardPure state1
    let (card2, _) = takeCardPure state2
    
    Assert.Equal(card1, card2)

[<Fact>]
let ``initGame should create correct initial state`` () =
    let game = initGame
    Assert.Equal(0, game.Score)
    Assert.Empty(game.CardsDrawn)
    Assert.False(game.IsOver)

[<Fact>]
let ``executeActionPure Stand should end game`` () =
    let initialState = initGame
    let randomState = initRandomState 42
    
    let (newState, _) = executeActionPure initialState Stand randomState
    
    Assert.True(newState.IsOver)
    Assert.Equal(0, newState.Score) // Score shouldn't change
    Assert.Empty(newState.CardsDrawn) // Cards shouldn't change

[<Fact>]
let ``executeActionPure Hit should add card and increase score`` () =
    let initialState = initGame
    let randomState = initRandomState 42
    
    let (newState, _) = executeActionPure initialState Hit randomState
    
    Assert.True(newState.Score > 0)
    Assert.Single(newState.CardsDrawn)
    Assert.Equal(newState.Score, List.sum newState.CardsDrawn)

[<Fact>]
let ``executeActionPure Hit with score over 21 should end game`` () =
    let highScoreState = { Score = 20; CardsDrawn = [10; 10]; IsOver = false }
    let randomState = initRandomState 42
    
    let (newState, _) = executeActionPure highScoreState Hit randomState
    
    // With score 20, any card 2-10 will bust (score > 21)
    if newState.Score > 21 then
        Assert.True(newState.IsOver)

[<Fact>]
let ``calculateReward should return 0 for bust`` () =
    let reward = calculateReward 22
    Assert.Equal(0.0, reward)

[<Fact>]
let ``calculateReward should return score for valid scores`` () =
    let reward1 = calculateReward 21
    let reward2 = calculateReward 15
    
    Assert.Equal(21.0, reward1)
    Assert.Equal(15.0, reward2)

[<Fact>]
let ``playGameWithSeed should be deterministic`` () =
    let records1 = playGameWithSeed 1 42
    let records2 = playGameWithSeed 1 42
    
    Assert.Equal(List.length records1, List.length records2)
    
    // First record should be identical
    if not (List.isEmpty records1) && not (List.isEmpty records2) then
        let firstRecord1 = List.head records1
        let firstRecord2 = List.head records2
        Assert.Equal(firstRecord1.GameId, firstRecord2.GameId)
        Assert.Equal(firstRecord1.Action, firstRecord2.Action)

[<Fact>]
let ``playGameWithSeed should produce valid PlayRecords`` () =
    let records = playGameWithSeed 1 42
    
    Assert.True(List.length records > 0)
    
    // Check first record
    let firstRecord = List.head records
    Assert.Equal(1, firstRecord.GameId)
    Assert.Equal(0, firstRecord.Step)
    Assert.Equal(0, firstRecord.CurrentScore)
    Assert.Empty(firstRecord.CardsDrawn)
    
    // Check that actions are valid (0 or 1)
    for record in records do
        Assert.True(record.Action = 0 || record.Action = 1)
        Assert.True(record.Step >= 0)
        Assert.True(record.ResultScore >= 0)

[<Fact>]
let ``playGameWithSeed should end when game is over`` () =
    let records = playGameWithSeed 1 42
    
    // Last record should have Done = true
    if not (List.isEmpty records) then
        let lastRecord = List.last records
        Assert.True(lastRecord.Done)
