import pandas as pd

def append_train_test(train, test):
    aggregations = {'Season' : 'first',
                    'Game_Date' : 'first',
                    'Away_Team' : 'first',
                    'Home_Team' : 'first',
                    'Rounded Viewers': 'sum'}
    train = train.groupby('Game_ID', as_index=False).agg(aggregations)
    train.columns.values[-1] = 'Total_Viewers'

    train['T'] = 'train'
    test['T'] = 'test'

    tt = train.append(test)

    return tt

def dummy_data(data, categorical_vars):
    for var in categorical_vars:
        dummies = pd.get_dummies(data[var], prefix = var, dummy_na=False)
        data = data.drop(var,1)
        data = pd.concat([data, dummies], axis=1)

    return data

def mape(actual, predict):
    n = len(predict)
    sum_score = 0.0
    for i in range(0,n):
        sum_score += abs(predict[i] - actual[i]) / float(actual[i])
        
    return (1 / float(len(predict))) * sum_score
