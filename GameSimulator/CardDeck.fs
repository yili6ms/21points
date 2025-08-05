module GameSimulator.CardDeck

open System

// Pure functional random number generation using a seed
type RandomState = { Seed: int; Next: int }

// Linear congruential generator for deterministic randomness
let private nextRandom (state: RandomState) =
    let a = 1664525
    let c = 1013904223
    let m = 2147483647
    let nextValue = (a * state.Seed + c) % m
    { Seed = nextValue; Next = nextValue }

// Generate a card value (1-10) from random state
let takeCardPure (randomState: RandomState) =
    let newState = nextRandom randomState
    let cardValue = abs newState.Next % 10 + 1
    cardValue, newState

// Initialize random state with a seed
let initRandomState seed = { Seed = seed; Next = seed }

// Impure wrapper for backward compatibility
let mutable private globalRandomState = initRandomState (int (DateTime.Now.Ticks % int64 Int32.MaxValue))

let takeCard () =
    let card, newState = takeCardPure globalRandomState
    globalRandomState <- newState
    card