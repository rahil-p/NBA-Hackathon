play_by_play  <- read.table('NBA Hackathon - Play by Play Data Sample (50 Games).txt', sep="", header=TRUE)
game_lineup   <- read.table('NBA Hackathon - Game Lineup Data Sample (50 Games).txt', sep="", header=TRUE)
event_codes   <- read.table('NBA Hackathon - Event Codes.txt', sep="", header=TRUE)


plus_minus    <- unique(game_lineup[c('Game_id', 'Person_id')])


play_by_play  <- play_by_play[order(play_by_play$Game_id,
                                    play_by_play$Period,
                                    -play_by_play$PC_Time,
                                    play_by_play$WC_Time,
                                    play_by_play$Event_Num),]
event_codes   <- event_codes[order(event_codes$Event_Msg_Type, 
                                   event_codes$Action_Type),]


for (game in unique(game_lineup$Game_id)) {
  
  lineups   <- game_lineup[game_lineup$Game_id == game,]        # df of starting lineups for each period in the game
  teams     <- unique(lineups$Team_id)                          # vector of the playing teams' IDs
  
  for (period in seq(1,4)) {  #CAN BE GREATER THAN 4 - find max period in the play by plays
    
    # df of lineups teams 1 and 2 (arbitrarily assigned) at the beginning of the current period
    current_lineup_1    <- factor(lineups[(lineups$Period == period) & (lineups$Team_id == teams[1]),]$Person_id)
    current_lineup_2    <- factor(lineups[(lineups$Period == period) & (lineups$Team_id == teams[2]),]$Person_id)
    print(current_lineup_1)
    # df of plays in the current period
    period_plays        <- play_by_play[(play_by_play$Game_id == game) & (play_by_play$Period == period),]

    for (play in 1:nrow(period_plays)) {
      event               <- period_plays[play, 'Event_Msg_Type']
        # 1   -   'Made shot'
        # 2   -   'Missed shot'
        # 3   -   'Free throw' (or 'No shot')
        # 4   -   'Rebound'
        # 5   -   'Turnover'
        # 6   -   'Foul'
        # 7   -   'Violation'
        # 8   -   'Substitution'
      action              <- period_plays[play, 'Action_Type']
      option1             <- period_plays[play, 'Option1']
      team                <- period_plays[play, 'Team_id']
      player1             <- period_plays[play, 'Person1']
      player2             <- period_plays[play, 'Person2']
      #player1team         <- ifelse(player1 %in% unique(game_lineup[game_lineup$Team_id == teams[1],]$Person_id), 1, ifelse(player1 %in% unique(game_lineup[game_lineup$Team_id == teams[2],]$Person_id), 2, 'Fuck'))
      #player2team         <- ifelse(player2 %in% unique(game_lineup[game_lineup$Team_id == teams[1],]$Person_id), 1, 2)
      
      if (event == 8) {                                       # 'Substitution'
        current_lineup_1 <- union(setdiff(current_lineup_1, player1), player2)
        
        print(player1)
        print(player2)
        print(current_lineup_1)
        
      } else if (event == 6) {                                # 'Foul'
        
      } else if (event == 3) {                                # 'Free throw'
        
      } else if (event == 1) {                                # 'Made shot'
        
      }
      
    }
  }
}























