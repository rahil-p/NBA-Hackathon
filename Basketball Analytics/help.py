import numpy as np

def get_period_starters(game_period_starters, period):
    game_teams = game_period_starters.Team_id.unique()

    period_starters = {}
    for team in game_teams:
        period_starters[team] = game_period_starters.loc[(game_period_starters.Period == period) &
                                                         (game_period_starters.Team_id == team)]['Person_id'].values
    return period_starters

def sync_lineup(period_lineup, game_stats):
    for value in period_lineup.values():
        for element in value:
            if element not in game_stats:
                game_stats[element] = 0
    return game_stats

def update_stats(game_stats, plus_team, minus_team, score):
    for player in plus_team:
        game_stats[player] += score
    for player in minus_team:
        game_stats[player] -= score

    return game_stats

def read_free_throw():
    pass

def free_throw(game_stats, team):
    pass #update_stats

def substitute(period_lineup, person1, person2):
    sub_team = None
    for key in period_lineup.keys():
        if person1 in period_lineup[key]:
            sub_team = key

    if sub_team is not None:
        period_lineup[sub_team] = np.setdiff1d(period_lineup[sub_team], np.array(person1))
        period_lineup[sub_team] = np.append(period_lineup[sub_team], person2)
        return period_lineup

    else:
        return "SUB_ERROR"
