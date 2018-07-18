from input import read_data, sort_pbp, sort_ec
from help import get_period_starters, sync_lineup, update_stats, free_throw, substitute
import pandas as pd
import numpy as np

def parse_games(gl, pbp):
    stats_dict = {}

    games = gl.Game_id.unique()
    for game in games:
        game_plays = sort_pbp(pbp.loc[pbp.Game_id == game])
        game_period_starters = gl.loc[gl.Game_id == game]
        print(stats_dict)
        stats_dict[game] = parse_periods(game_plays, game_period_starters, game_stats={})

    return stats_dict

def parse_periods(game_plays, game_period_starters, game_stats):
    for period in range(0, game_plays.Period.max() + 1):
        period_plays = sort_pbp(game_plays.loc[game_plays.Period == period])
        period_lineup = get_period_starters(game_period_starters, period)

        game_stats = parse_plays(period_plays, period_lineup, game_stats)
    return game_stats

def parse_plays(period_plays, period_lineup, game_stats):
    for i, play in period_plays.iterrows():
        event = play.Event_Msg_Type
        action = play.Action_Type
        option1 = play.Option1

        team = play.Team_id
        for key in period_lineup:
            if key != team:
                other_team = key

        person1 = play.Person1
        person2 = play.Person2

        game_stats = sync_lineup(period_lineup, game_stats)

        if (event == 3) & (action != 0):        # free throw
            pass
            #parse the free throw description
            #update stats


        elif event == 1:                        # made shot
            game_stats = update_stats(game_stats,
                                      plus_team=period_lineup[team],
                                      minus_team=period_lineup[other_team],
                                      score=option1)

        elif event == 8:                        # substitution
        #if appropriate:
            period_lineup = substitute(period_lineup, person1, person2)

        elif event == 6:                        # foul
            pass


            #pause substitution until the completion of n free throws (if free throws awarded)

    return game_stats





def main():
    pbp, gl, ec = read_data({'pbp': 'NBA Hackathon - Play by Play Data Sample (50 Games).txt',
                             'gl':  'NBA Hackathon - Game Lineup Data Sample (50 Games).txt',
                             'ec':  'NBA Hackathon - Event Codes.txt'})
    ec = sort_ec(ec)

    parse_games(gl, pbp)

if __name__ == '__main__':
    main()
