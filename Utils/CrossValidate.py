import numpy as np
import pandas as pd 
import lightgbm as lgb
import gc
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

ROUTE = '/Users/davidlee/python/TBrain/data/'

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

class CrossValidate():
    def __init__(self):
        return None
    
    def sliding_window(self, X, y, cat, boost_round=500, n_fold=3, random_seed=6):
        '''
        This is the implementation of sliding window cross validation.
        ==================================
        | train| test |      |      |
        |      | train| test |      |
        |      |      | train| test |
        ==================================
        X: train data
        y: train label
        cat: categorical list
        '''
        shape_fold = int(X.shape[0] / (n_fold + 1))
        shape_list = []
        for i in range(n_fold):
            shape_list.append(shape_fold * (i + 1))
        Xs = np.split(X, shape_list)
        ys = np.split(y, shape_list)
        params = {
            'objective': 'binary',
            'early_stopping_rounds': 100,
            'learning_rate': 0.01,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'max_depth': -1,
            'num_leaves': 100,
            'seed': random_seed
        }
        res = []
        for i in range(n_fold):
            eval_dict = {}
            train_data = lgb.Dataset(data=Xs[i], label=ys[i], categorical_feature=cat)
            val_data = lgb.Dataset(data=Xs[i], label=ys[i], reference=train_data, categorical_feature=cat)
            lgb.train(params, 
                train_data,
                valid_sets=[val_data], 
                evals_result=eval_dict,
                num_boost_round=boost_round,
                verbose_eval=50, 
                feval=lgb_f1_score)
            res.append(max(eval_dict['valid_0']['f1']))
            del eval_dict, train_data, val_data
            gc.collect()
        del Xs, ys
        gc.collect()
        return res
    
    def expanding_window(self, X, y, cat, boost_round=500, n_fold=3, random_seed=6, last_fold=3):
        '''
        This is the implementation of expanding window cross validation.
        ==================================
        | train| test |      |      |
        |   train     | test |      |
        |       train        | test |
        ==================================
        X: train data
        y: train label
        cat: categorical list
        last_fold: the last folds want to run (> 0)
        '''
        tscv = TimeSeriesSplit(n_fold)
        res = []
        params = {
            'objective': 'binary',
            'early_stopping_rounds': 100,
            'learning_rate': 0.01,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'max_depth': -1,
            'num_leaves': 100,
            'seed': random_seed
        }
        
        count = 1
        i = 0
        for train_index, val_index in tscv.split(X):
            if(i < n_fold - last_fold):
                i += 1
                continue
            print('\n' + '='*40 + '\n' + '[ CV Round {} ]'.format(count))
            i += 1
            count += 1
            X_tr, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
            y_tr, y_val = y[train_index], y[val_index]
            train_data = lgb.Dataset(data=X_tr, label=y_tr, categorical_feature=cat)
            val_data = lgb.Dataset(data=X_val, label=y_val, reference=train_data, categorical_feature=cat)
            eval_dict = {}
            lgb.train(params, 
                train_data,
                valid_sets=[val_data], 
                evals_result=eval_dict,
                num_boost_round=boost_round,
                verbose_eval=50, 
                feval=lgb_f1_score)
            res.append(max(eval_dict['valid_0']['f1']))
            del eval_dict, train_data, val_data
            gc.collect()
            print('\n' + '='*40 + '\n')
        return res

    def ratio_window(self, X, y, cat, boost_round=500, random_seed=6, last_fold=3):
        '''
        Since the train test ratio is 3: 1 from time perspective, the validation should be 1/3 of train set too.
        X: train data
        y: train label
        cat: categorical list
        '''
        shape_fold = int(X.shape[0] / 3)
        shape_list = []
        for i in range(2):
            shape_list.append(shape_fold * (i + 1))
        shape_list.append(X.shape[0])
        params = {
            'objective': 'binary',
            'early_stopping_rounds': 100,
            'learning_rate': 0.01,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'max_depth': -1,
            'num_leaves': 100,
            'seed': random_seed
        }
        res = []
        for i in range(3 - last_fold, 3):
            eval_dict = {}
            shape_val = int(shape_list[i] * 3 / 4)
            train_data = lgb.Dataset(data=X.iloc[:shape_val, :], label=y[:shape_val], categorical_feature=cat)
            val_data = lgb.Dataset(data=X.iloc[shape_val:shape_list[i], :], label=y[shape_val:shape_list[i]], reference=train_data, categorical_feature=cat)
            lgb.train(params, 
                train_data,
                valid_sets=[val_data], 
                evals_result=eval_dict,
                num_boost_round=boost_round,
                verbose_eval=50, 
                feval=lgb_f1_score)
            res.append(max(eval_dict['valid_0']['f1']))
            del eval_dict, train_data, val_data
            gc.collect()
        return res

    
    def add_cv_test(self, X, y, cat, boost_round=500, n_fold=3, random_seed=6):
        cv_test = pd.read_csv(ROUTE + 'cv_test.csv')
        cv_list = {}
        cv_list['feature_num'] = X.shape[1]

        print('slide...')
        res = self.sliding_window(X, y, cat, boost_round=boost_round, n_fold=n_fold, random_seed=random_seed)
        cv_list['slide'] = sum(res) / len(res)

        print('expand...')
        res = self.expanding_window(X, y, cat, boost_round=boost_round, n_fold=n_fold, random_seed=random_seed)
        cv_list['expand'] = sum(res) / len(res)
        cv_list['expand_last2'] = sum(res[1:]) / len(res[1:])

        print('ratio...')
        res = self.ratio_window(X, y, cat, boost_round=boost_round, random_seed=random_seed)
        cv_list['ratio'] = sum(res) / len(res)
        cv_list['ratio_last2'] = sum(res[1:]) / len(res[1:])
        cv_list['public'] = 0

        cv_test = cv_test.append(cv_list, ignore_index=True)
        cv_test.to_csv(ROUTE + 'cv_test.csv', index=False)
        del cv_test, cv_list
        gc.collect()
        return None

    def add_cv_public(self, score, feature_num=None):
        '''
        if feature_num assigned, will try to find it by using feature_num
        otherwise, the last entry is updated
        '''
        cv_test = pd.read_csv(ROUTE + 'cv_test.csv')
        if(feature_num):
            cv_test.loc[(cv_test['feature_num'] == feature_num), 'public'] = score
        else:
            cv_test.loc[cv_test.shape[0] - 1, 'public'] = score
        cv_test.to_csv(ROUTE + 'cv_test.csv', index=False)
        return None
