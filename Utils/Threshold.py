import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

from Utils.CrossValidate import lgb_f1_score

def get_f1_score(threshold, y_true, y_proba):
    y_pred = np.where(y_proba >= threshold, 1, 0)
    return f1_score(y_true, y_pred)

class Thresehold():
    def __init__(self):
        return None

    def threshold_search(self, y_true, y_proba):
        '''
        Using true label and probabily prediction from the model, 
        searching for the threshold taht maximize the f1_score

        Parameters: 
        y_true: true label
        y_proba: the probability from the model

        Return:
        search_result: dict of best threshold and it's best score
        '''
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        thresholds = np.append(thresholds, 1.001) 
        F = 2 / (1 / precision + 1 / recall) 
        best_score = np.max(F)
        best_th = thresholds[np.argmax(F)]
        search_result = {'threshold': best_th , 'f1': best_score}
        return search_result 

    def calc_threshold_diff(self, X, y, cat, n_fold, boost_round=100):
        '''
        Use expanding window method to record each threshold difference from the fold's best f1 score.

        Parameters: 
        X: training data
        y: training label
        cat: categorical list for the lgb model
        n_fold: fold number
        boost_round: boosting round for the model

        Return:
        df: DataFrame with each threshold difference value from the fold's best f1 score.
        '''
        tscv = TimeSeriesSplit(n_fold)
        params = {
            'objective': 'binary',
            'early_stopping_rounds': 100,
            'learning_rate': 0.01,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'max_depth': -1,
            'num_leaves': 100,
            'metric': 'None',
            'seed': 6
        }
        try_threshold = np.linspace(0.001, 0.999, 999)
        df = pd.DataFrame({'threshold': try_threshold})
        i = 0
        for train_index, val_index in tscv.split(X):
            train_data = lgb.Dataset(data=X.loc[train_index, :], label=y[train_index], categorical_feature=cat)
            val_data = lgb.Dataset(data=X.loc[val_index, :], label=y[val_index], reference=train_data, categorical_feature=cat)
            clf = lgb.train(
                params, 
                train_data,
                valid_sets=[val_data],
                num_boost_round=boost_round,
                verbose_eval=0, 
                feval=lgb_f1_score)
            # get best result f1 for val set
            prob = clf.predict(X.loc[val_index, :])
            search_result = self.threshold_search(y[val_index], prob)
            
            # add result
            i += 1
            df['fold_' + str(i)] = df['threshold'].apply(lambda x: get_f1_score(x, y[val_index], prob))
            df['fold_' + str(i)] = df['fold_' + str(i)] - search_result['f1']
            print(f'{i} fold run: search threshold {search_result}')
        return df

    def get_best_threshold(self, df):
        '''
        Within the range 0.05 from the min f1 score threshold, find the best threshold with the min std value
        
        Parameters:
        df: the DataFrame from calc_threshold_diff method

        Return:
        best_threshold: the best threshold value
        '''
        min_val = df.set_index('threshold').mean(axis=1).argmax()
        best_threshold = df[(df['threshold'] >= min_val - 0.05)&(df['threshold'] <= min_val + 0.05)].set_index('threshold').std(axis=1).argmin()
        return best_threshold