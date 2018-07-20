from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
import functools, itertools, operator
import time


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

def split_train_test(tt):
    train = tt.loc[tt['T'] == 'train']
    test = tt.loc[tt['T'] == 'test']

    return [train, test]

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

def suggest_hyperparameters(model_type,
                            training_set, validation_set,
                            predictor_name,
                            scorer,
                            switch=1,
                            **learning_parameters):
    if switch == 1:
        hp_names = list(learning_parameters.keys())
        hp_combos = itertools.product(*list(learning_parameters.values()))
        hp_length = functools.reduce(operator.mul, [len(x) for x in list(learning_parameters.values())], 1)

        all_model_parameters = []
        for i, hyperparameter in enumerate(hp_combos):

            start_time = time.time()

            model_parameters = dict(zip(hp_names, hyperparameter))
            _, _, metric, train_metric = fit_predict_assess(model_type=RandomForestRegressor,
                                                            training_set=training_set,
                                                            validation_set=validation_set,
                                                            predictor_name='Total_Viewers',
                                                            scorer=mape,
                                                            **model_parameters)
            model_parameters['Train_Accuracy'] = train_metric
            model_parameters['Validation_Accuracy'] = metric
            model_parameters['Overfitting_Distance'] = abs(train_metric-metric)

            all_model_parameters.append(model_parameters)

            end_time = time.time()

            if i < 1:
                seconds = end_time - start_time
                print('The first of %d iterations took %d seconds; expect a total of %d seconds' % (hp_length,
                                                                                                    seconds,
                                                                                                    hp_length * seconds))
                break_or_continue = input('Enter Y to continue or N to cancel hyperparameter suggestion: ')
                if break_or_continue in ['y', 'Y']:
                    continue
                else:
                    return

        hp_df = pd.DataFrame(all_model_parameters)

        return hp_df

def fit_predict_assess(model_type,
                       training_set, validation_set,
                       predictor_name,
                       scorer,
                       **model_parameters):
    # fit
    model = model_type(**model_parameters)
    model.fit(training_set.iloc[:,3:len(training_set.columns)], training_set[predictor_name])

    # predict
    predictions = model.predict(validation_set.iloc[:,3:len(validation_set.columns)])
    train_predictions = model.predict(training_set.iloc[:,3:len(training_set.columns)])     # not returned at the moment

    # assess
    metric = scorer(np.asarray(validation_set[predictor_name]), np.asarray(predictions))
    train_metric = scorer(np.asarray(training_set[predictor_name]), np.asarray(train_predictions))

    return model, predictions, metric, train_metric
