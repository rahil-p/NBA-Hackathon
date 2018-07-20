from processing import read_data, pickle_dump
from ml import append_train_test, dummy_data, mape, fit_predict_assess, suggest_hyperparameters
from scrape import get_driver

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime as dt


def add_features(train, test, game_data, player_data, features, categorical_features, driver):
    tt = append_train_test(train, test)
    tt = game_features(tt, game_data, driver)

    tt = tt[features]
    tt = dummy_data(tt, categorical_features)

    teams = tt.Home_Team.unique()
    for team in teams:
        tt[team] = ((tt.Home_Team == team) | (tt.Away_Team == team)) * 1

    tt.drop(['Home_Team','Away_Team'], axis=1, inplace=True)

    train = tt.loc[tt['T'] == 'train']
    test = tt.loc[tt['T'] == 'test']

    return [train, test]

def game_features(tt, game_data, driver):
    games = game_data.Game_ID.unique()
    tt = game_time_features(tt)
    # game_scrape(tt, game_data, driver)
    tt = game_data_features(tt, game_data, games)

    return tt

def game_time_features(tt):
    tt[['Month', 'Date', 'Year']] = tt.Game_Date.str.split('/', expand=True)
    tt['Time_Stamp'] = pd.to_datetime(tt.Game_Date, infer_datetime_format=True)
    tt['Day_Of_Week'] = tt.Time_Stamp.dt.dayofweek        # Monday = 0, Sunday = 6

    return tt

def game_data_features(tt, game_data, games):
    game_daily_count = game_data.groupby('Game_Date', as_index=False)['Game_ID'].count()

    aggregations = {'Qtr_4_Score' : lambda x: (max(x)-min(x))/min(x),
                    'L2M_Score' : lambda x: (max(x)-min(x))/min(x),
                    'Final_Score' : lambda x: (max(x)-min(x))/min(x),
                    'Lead_Changes': lambda x: sum(x),
                    'Largest_Lead': lambda x: max(x),
                    'Wins_Entering_Gm': lambda x: max(x)-min(x),
                    'Losses_Entering_Gm': lambda x: max(x)-min(x),
                    'Ties': lambda x: sum(x),
                    'Team_Minutes': lambda x: max(x)}
    game_diffs = game_data.groupby('Game_ID', as_index=False).agg(aggregations)

    tt = game_diffs_fill_append(tt, game_data, game_diffs, game_daily_count)

    return tt

def game_diffs_fill_append(tt, game_data, game_diffs, game_daily_count):
    aggregations = {'Team' : lambda x: tuple(x)}
    aggregations2 = {'Game_ID' : lambda x: tuple(x)}
    similar_games = game_data.groupby('Game_ID',
                                      as_index=False).agg(aggregations).groupby('Team',
                                                                                as_index=False).agg(aggregations2)

    for i, game_row in game_diffs.iterrows():
        if pd.isnull(game_row).any():
            for _, teams_row in similar_games.iterrows():
                if game_row.Game_ID in teams_row.Game_ID:
                    fill_games = np.setdiff1d(teams_row.Game_ID, np.array(game_row.Game_ID))
                    fill_data = game_diffs[game_diffs['Game_ID'].isin(fill_games)].mean(axis=0, skipna=True)
                    game_diffs.loc[i, 'Qtr_4_Score'] = fill_data[1]
                    game_diffs.loc[i, 'L2M_Score'] = fill_data[2]
                    game_diffs.loc[i, 'Final_Score'] = fill_data[3]
                    game_diffs.loc[i, 'Lead_Changes'] = fill_data[4]
                    game_diffs.loc[i, 'Largest_Lead'] = fill_data[5]
                    game_diffs.loc[i, 'Wins_Entering_Gm'] = fill_data[6]
                    game_diffs.loc[i, 'Losses_Entering_Gm'] = fill_data[7]
                    game_diffs.loc[i, 'Ties'] = fill_data[8]
                    game_diffs.loc[i, 'Team_Minutes'] = fill_data[9]

    for col in ['Qtr_4_Score', 'L2M_Score', 'Final_Score', 'Lead_Changes', 'Largest_Lead',
                'Wins_Entering_Gm', 'Losses_Entering_Gm', 'Ties', 'Team_Minutes']:
        game_diffs[col].fillna((game_diffs[col].mean()), inplace=True)

    tt['Q4_Diff'] = tt['Game_ID'].map(game_diffs.set_index('Game_ID')['Qtr_4_Score'])
    tt['L2M_Diff'] = tt['Game_ID'].map(game_diffs.set_index('Game_ID')['L2M_Score'])
    tt['Final_Diff'] = tt['Game_ID'].map(game_diffs.set_index('Game_ID')['Final_Score'])
    tt['Lead_Changes_Sum'] = tt['Game_ID'].map(game_diffs.set_index('Game_ID')['Lead_Changes'])
    tt['Largest_Lead_Max'] = tt['Game_ID'].map(game_diffs.set_index('Game_ID')['Largest_Lead'])
    tt['Wins_Diff'] = tt['Game_ID'].map(game_diffs.set_index('Game_ID')['Wins_Entering_Gm'])
    tt['Losses_Diff'] = tt['Game_ID'].map(game_diffs.set_index('Game_ID')['Losses_Entering_Gm'])
    tt['Ties_Total'] = tt['Game_ID'].map(game_diffs.set_index('Game_ID')['Ties'])
    tt['Team_Minutes_Total'] = tt['Game_ID'].map(game_diffs.set_index('Game_ID')['Team_Minutes'])
    tt['Today_Games'] = tt['Game_Date'].map(game_daily_count.set_index('Game_Date')['Game_ID'])

    return tt

def team_features(tt, game_data, games):
    pass

def player_features():
    pass

def player_data_features():
    pass



def main():
    # for web sraping
    driver = get_driver('/chromedriver', switch=0)  # turned on/off with switch

    # reading data
    train, test, player_data, game_data = read_data({'train' : '../data/training_set.csv',
                                                     'test' : '../data/test_set.csv',
                                                     'player_data' : '../data/player_data.csv',
                                                     'game_data' : '../data/game_data.csv'})
    # saving original test data for appending predictions later
    test_output = test.copy()

    # define features
    features = ['T','Total_Viewers','Game_ID','Home_Team','Away_Team','Season','Month', 'Day_Of_Week',
                'Q4_Diff','L2M_Diff','Final_Diff','Lead_Changes_Sum','Largest_Lead_Max','Wins_Diff','Losses_Diff',
                'Ties_Total','Team_Minutes_Total','Today_Games']

    categorical_features = ['Season', 'Month', 'Day_Of_Week']

    # split
    train, test = add_features(train, test, game_data, player_data, features, categorical_features, driver)
    training, validation = train_test_split(train, test_size=.2)

    # suggest hyperparameters
    hp_suggestions = suggest_hyperparameters(model_type=RandomForestRegressor,
                                             training_set=training, validation_set=validation,
                                             predictor_name='Total_Viewers',
                                             scorer=mape,
                                             switch=1,
                                             n_estimators=[1000],
                                             max_features=['auto'],
                                             max_depth=list(range(9,13,1)),
                                             min_samples_split=list(range(20,35,5)),
                                             min_samples_leaf=list(range(1,6,1)))

    if isinstance(hp_suggestions, pd.DataFrame):
        hp_suggestions.to_csv('../data/hp_suggestions.csv', index=False)

    # fit, predict, assess
    model, predictions, metric, train_metric = fit_predict_assess(model_type=RandomForestRegressor,
                                                                  training_set=training, validation_set=validation,
                                                                  predictor_name='Total_Viewers',
                                                                  scorer=mape,
                                                                  n_estimators=1000,
                                                                  max_features='auto',
                                                                  max_depth=12,
                                                                  min_samples_split=20,
                                                                  min_samples_leaf=3)


    # plot
    validation.hist(column='Total_Viewers', bins=40, alpha=.2)
    pd.DataFrame({'Predictions' : predictions}).hist(column='Predictions', bins=40, alpha=.2, color='r')

    plt.show()


    # final predict
    train_decisions = model.predict(training.iloc[:,3:len(validation.columns)])
    test_decisions = model.predict(test.iloc[:,3:len(validation.columns)])

    test_output['Total_Viewers'] = test_decisions

    print(metric)
    print(train_metric)
    print(metric - train_metric)

    pickle_dump([["../data/test.pickle", test_output]], switch=1)

if __name__ == '__main__':
    main()
