open System
open GameSimulator.Types
open GameSimulator.Game
open GameSimulator.PlayLogger

[<EntryPoint>]
let main argv =
    let numGames = 
        if argv.Length > 0 then 
            Int32.Parse(argv.[0])
        else 
            10000
    
    printfn "Starting simulation of %d games..." numGames
    
    let mutable allRecords = []
    let startTime = DateTime.Now
    
    for gameId in 1 .. numGames do
        let gameRecords = playGame gameId
        allRecords <- allRecords @ gameRecords
        
        if gameId % 1000 = 0 then
            printfn "Completed %d games..." gameId
    
    let endTime = DateTime.Now
    let elapsed = endTime - startTime
    
    writeToCsv "gameplay_log.csv" allRecords
    writeToJson "gameplay_log.json" allRecords
    
    let totalGames = allRecords |> List.map (fun r -> r.GameId) |> List.distinct |> List.length
    let avgStepsPerGame = float allRecords.Length / float totalGames
    let avgReward = 
        allRecords 
        |> List.filter (fun r -> r.Done) 
        |> List.averageBy (fun r -> r.Reward)
    
    printfn "\n=== Simulation Complete ==="
    printfn "Total games: %d" totalGames
    printfn "Total records: %d" allRecords.Length
    printfn "Average steps per game: %.2f" avgStepsPerGame
    printfn "Average reward: %.2f" avgReward
    printfn "Time elapsed: %.2f seconds" elapsed.TotalSeconds
    
    0
