module ActionPredictor.DataLoader

open System.IO
open FSharp.Data
open Types

let parseCardsDrawn (cardsStr: string) =
    if System.String.IsNullOrWhiteSpace(cardsStr) then
        []
    else
        cardsStr.Split(';') 
        |> Array.filter (fun s -> not (System.String.IsNullOrWhiteSpace(s)))
        |> Array.map int 
        |> Array.toList

let loadFromCsv (filename: string) =
    let lines = File.ReadAllLines(filename)
    
    if lines.Length <= 1 then
        []
    else
        lines
        |> Array.skip 1
        |> Array.map (fun line ->
            let parts = line.Split(',')
            {
                GameId = int parts.[0]
                Step = int parts.[1]
                CurrentScore = int parts.[2]
                CardsDrawn = parseCardsDrawn (parts.[3].Replace("\"", ""))
                Action = int parts.[4]
                ResultScore = int parts.[5]
                Reward = float parts.[6]
                Done = bool.Parse(parts.[7])
            })
        |> Array.toList

let printStats (records: PlayRecord list) =
    let totalRecords = records.Length
    let totalGames = records |> List.map (fun r -> r.GameId) |> List.distinct |> List.length
    let hitActions = records |> List.filter (fun r -> r.Action = 1) |> List.length
    let standActions = records |> List.filter (fun r -> r.Action = 0) |> List.length
    
    printfn "=== Data Statistics ==="
    printfn "Total records: %d" totalRecords
    printfn "Total games: %d" totalGames
    printfn "Hit actions: %d (%.1f%%)" hitActions (float hitActions * 100.0 / float totalRecords)
    printfn "Stand actions: %d (%.1f%%)" standActions (float standActions * 100.0 / float totalRecords)