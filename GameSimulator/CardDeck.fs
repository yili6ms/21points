module GameSimulator.CardDeck

open System

let private random = Random()

let takeCard () =
    random.Next(1, 11)