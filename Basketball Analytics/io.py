from processing import read_data, sort_pbp, sort_ec, get_output
from parse import get_period_starters, sync_lineup, \
                  get_lag_events, assess_lag, clear_queue, update_stats, \
                  substitute, assess_ft_completion, assess_ft_completion_correction
import pandas as pd
import numpy as np

def parse_games(gl, pbp):
    stats_dict = {}

    games = gl.Game_id.unique()
    for game in games:
        game_plays = sort_pbp(pbp.loc[pbp.Game_id == game])
        game_period_starters = gl.loc[gl.Game_id == game]

        #print(game)
        stats_dict[game] = parse_periods(game_plays, game_period_starters, game, game_stats={})

    return stats_dict

def parse_periods(game_plays, game_period_starters, game, game_stats):
    for period in range(0, game_plays.Period.max() + 1):
        period_plays = sort_pbp(game_plays.loc[game_plays.Period == period])
        period_lineup = get_period_starters(game_period_starters, period)

        #print(period)
        game_stats = parse_plays(period_plays, period_lineup, game, game_stats)

    return game_stats

def parse_plays(period_plays, period_lineup, game, game_stats):
    period_plays = period_plays.reset_index(drop=True)
    queued_sub, queue_sub = clear_queue()

    for i, play in period_plays.iterrows():
        event = play.Event_Msg_Type
        action = play.Action_Type
        option1 = play.Option1
        team = play.Team_id
        other_team = [x for x in list(period_lineup.keys()) if x != team][0]
        person1 = play.Person1
        person2 = play.Person2

        game_stats = sync_lineup(period_lineup, game_stats)

        # free throw
        if (event == 3) & (action != 0):
            if queue_sub == True:
                game_stats = update_stats(game_stats,
                                          plus_team=period_lineup[team],
                                          minus_team=period_lineup[other_team],
                                          score=option1)

                period_lineup, action, queued_sub, queue_sub = assess_ft_completion(period_lineup, action,
                                                                                    queued_sub, queue_sub)

            elif queue_sub == False:
                # ad hoc solutions (more robust options?)
                period_lineup, game_stats = assess_ft_completion_correction(period_lineup, game_stats,
                                                                            team, other_team,
                                                                            game, action, option1)

        # made shot
        elif event == 1:
            game_stats = update_stats(game_stats,
                                      plus_team=period_lineup[team],
                                      minus_team=period_lineup[other_team],
                                      score=option1)

        # substitution
        elif event == 8:
            if queue_sub == False:
                period_lineup = substitute(period_lineup, person1, person2)
            else:
                queued_sub = substitute(period_lineup, person1, person2)

        # foul
        elif event == 6:
            lag_events = get_lag_events(i, period_plays)
            queue_sub = assess_lag(lag_events)

            #pause substitution until the completion of n free throws (if free throws awarded)

    return game_stats


def main():
    pbp, gl, ec = read_data({'pbp': 'NBA Hackathon - Play by Play Data Sample (50 Games).txt',
                             'gl':  'NBA Hackathon - Game Lineup Data Sample (50 Games).txt',
                             'ec':  'NBA Hackathon - Event Codes.txt'})
    ec = sort_ec(ec)
    #pbp = sort_pbp(pbp)

    stats_dict = parse_games(gl, pbp)
    output_df = get_output(stats_dict)

    output_df.to_csv('Crew_Q1_BBALL.csv', index=False)

if __name__ == '__main__':
    main()
