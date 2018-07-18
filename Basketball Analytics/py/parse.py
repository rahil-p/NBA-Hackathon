import numpy as np

def get_period_starters(game_period_starters, period):
    game_teams = game_period_starters.Team_id.unique()

    period_starters = {}
    for team in game_teams:
        period_starters[team] = game_period_starters.loc[(game_period_starters.Period == period) &
                                                         (game_period_starters.Team_id == team)]['Person_id'].values

    return period_starters

def sync_lineup(period_lineup, game_stats):
    for values in period_lineup.values():
        for element in values:
            if element not in game_stats:
                game_stats[element] = 0

    return game_stats

def get_lag_events(i, period_plays, lag=5):
    lag_events = []
    for j in range(i, min(i + lag, len(period_plays))):
        lag_event = period_plays.iloc[j].Event_Msg_Type
        lag_events.append(lag_event)
    return lag_events

def assess_lag(lag_events):
    for event in lag_events[1:]:
        if event == 3:
            return True
        elif event == 6:
            return False

def clear_queue():
    return [None, False]

def update_stats(game_stats, plus_team, minus_team, score):
    for player in plus_team:
        game_stats[player] += score
    for player in minus_team:
        game_stats[player] -= score

    return game_stats

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

def assess_ft_completion(period_lineup, action, queued_sub, queue_sub):
    if action in [10, 12, 15, 16, 17, 19, 22, 26, 29]:
        if queued_sub is not None:
            period_lineup = queued_sub
            queued_sub, queue_sub = clear_queue()
    elif action not in [11, 13, 14, 18, 20, 21, 25, 27, 28]:
        print('UNKNOWN FT ACTION')

    return [period_lineup, action, queued_sub, queue_sub]

def assess_ft_completion_correction(period_lineup, game_stats,
                                    team, other_team,
                                    game, action, option1):
    if (game == '021fd159b55773fba8157e2090fe0fe2') & (action in [11,12]):
        # no intermediate substitutions
        game_stats = update_stats(game_stats,
                                  plus_team=period_lineup[team],
                                  minus_team=period_lineup[other_team],
                                  score=option1)


    elif (game == 'c18a10de1375b1f12aa17ef6cc540102') & (action == 16):
        # 1 intermediate substitution (in - , out - )
        period_lineup = substitute(period_lineup,
                                   '952cb62f00fbb58407f3a7cd89c3a7bf',
                                   'b248b1d9caa41a3562d67584c1a0b399')
        game_stats = update_stats(game_stats,
                                  plus_team=period_lineup[team],
                                  minus_team=period_lineup[other_team],
                                  score=option1)
        period_lineup = substitute(period_lineup,
                                   'b248b1d9caa41a3562d67584c1a0b399',
                                   '952cb62f00fbb58407f3a7cd89c3a7bf')
    else:
        print('UNEXPECTED FT')

    return [period_lineup, game_stats]
