module GameSimulator.PlayLogger

open System.IO
open Types
open FSharp.Data

let private cardsToString (cards: int list) =
    cards |> List.map string |> String.concat ";"

let writeToCsv (filename: string) (records: PlayRecord list) =
    use writer = new StreamWriter(filename)
    
    writer.WriteLine("GameId,Step,CurrentScore,CardsDrawn,Action,ResultScore,Reward,Done")
    
    for record in records do
        let line = 
            sprintf "%d,%d,%d,\"%s\",%d,%d,%.1f,%b"
                record.GameId
                record.Step
                record.CurrentScore
                (cardsToString record.CardsDrawn)
                record.Action
                record.ResultScore
                record.Reward
                record.Done
        writer.WriteLine(line)
    
    printfn "Wrote %d records to %s" records.Length filename

let writeToJson (filename: string) (records: PlayRecord list) =
    let json = 
        records
        |> List.map (fun r -> 
            sprintf """{"GameId":%d,"Step":%d,"CurrentScore":%d,"CardsDrawn":[%s],"Action":%d,"ResultScore":%d,"Reward":%.1f,"Done":%b}"""
                r.GameId
                r.Step
                r.CurrentScore
                (r.CardsDrawn |> List.map string |> String.concat ",")
                r.Action
                r.ResultScore
                r.Reward
                r.Done)
        |> String.concat ",\n"
    
    let fullJson = sprintf "[\n%s\n]" json
    File.WriteAllText(filename, fullJson)
    printfn "Wrote %d records to %s" records.Length filename