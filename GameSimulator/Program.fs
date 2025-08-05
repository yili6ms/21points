open System
open GameSimulator.Types
open GameSimulator.Game
open GameSimulator.PlayLogger

// Pure functional simulation runner
let runSimulation (numGames: int) =
    let startTime = DateTime.Now
    
    // Generate all games functionally with different seeds
    let allRecords = 
        [1..numGames]
        |> List.mapi (fun i gameId -> 
            let seed = gameId * 42 + i  // Deterministic but varied seeds
            playGameWithSeed gameId seed)
        |> List.concat
    
    let endTime = DateTime.Now
    let elapsed = endTime - startTime
    
    // Calculate statistics functionally
    let stats = {|
        TotalGames = allRecords |> List.map (fun r -> r.GameId) |> List.distinct |> List.length
        TotalRecords = allRecords.Length
        AvgStepsPerGame = float allRecords.Length / float numGames
        AvgReward = 
            allRecords 
            |> List.filter (fun r -> r.Done) 
            |> List.averageBy (fun r -> r.Reward)
        ElapsedSeconds = elapsed.TotalSeconds
    |}
    
    (allRecords, stats)

// Pure functional progress reporter
let reportProgress (gameId: int) =
    if gameId % 1000 = 0 then
        printfn "Completed %d games..." gameId

// Pure functional statistics printer
let printStats (stats: {| TotalGames: int; TotalRecords: int; AvgStepsPerGame: float; AvgReward: float; ElapsedSeconds: float |}) =
    printfn "\n=== Simulation Complete ==="
    printfn "Total games: %d" stats.TotalGames
    printfn "Total records: %d" stats.TotalRecords
    printfn "Average steps per game: %.2f" stats.AvgStepsPerGame
    printfn "Average reward: %.2f" stats.AvgReward
    printfn "Time elapsed: %.2f seconds" stats.ElapsedSeconds

[<EntryPoint>]
let main argv =
    let numGames = 
        if argv.Length > 0 then 
            Int32.Parse(argv.[0])
        else 
            10000
    
    printfn "Starting simulation of %d games..." numGames
    
    // Run simulation functionally
    let (allRecords, stats) = runSimulation numGames
    
    // Side effects only at the end
    writeToCsv "gameplay_log.csv" allRecords
    writeToJson "gameplay_log.json" allRecords
    
    printStats stats
    
    0
