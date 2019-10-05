import os
import sys
import warnings
import gc
import random
from argparse import ArgumentParser
sys.path.append('..')
warnings.simplefilter('ignore')

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
# import category_encoders as ce

from Utils.Feature import FeatureEngineer
from Utils.CrossValidate import CrossValidate
from Utils.read_data import read
from Utils.func import train_submit
from Utils.Threshold import Threshold
# from Utils.plot import plot_dist_diff, plot_high_fraud, plot_high_countfraud


def get_dataset():
    li = ['acquirer', 'bank', 'card', 'money', 'trade_cat', 'coin', 'online', 'trade_type',\
        'fallback', '3ds', 'fraud_ind', 'pay_type', 'install', 'term', 'date', 'time', 'mcc', 'shop', 'excess',\
        'city', 'nation', 'status', 'txkey']

    # combine = read('../data/combine_mr.pkl')
    train = read('./data/train_mr.pkl')
    test = read('./data/test_mr.pkl')

    combine = pd.concat([train, test])
    combine = combine.reset_index(drop=True)
    combine = combine[li]   # reset dataframe column order (affect : LGBM sort by column index)
    combine = combine.reset_index(drop=True)
    return combine


def train(action='cv', file_name='submit001', feature='new', feature_fname='feature_ver1l', n_fold=5):
    TRAIN_SHAPE = 1521787
    not_train = ['txkey', 'date', 'time', 'fraud_ind']
    need_encode = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation']
    cat = ['status', 'trade_cat', 'pay_type', 'trade_type']
    feature_root = os.path.join('.', 'data', 'feature')
    os.makedirs(feature_root, exist_ok=True)

    # 1. pre process
    print('\n[Step 1/3] Start Feature Engineer Pre-processing ... \n')
    feature_path = os.path.join(feature_root, feature_fname+'.pkl')
    if feature == 'new':
        # get dataset
        dataset = get_dataset()
        preprocessor = FeatureEngineer()
        preprocessor.engineer_all(dataset)
        with open(feature_path, 'wb') as file:
            pickle.dump(dataset, file)
    elif feature == 'load':
        with open(feature_path, 'rb') as file:
            dataset = pickle.load(file)

    # 2. calculate threshold
    print('\n[Step 2/3] Calculate best threshold ... \n')
    
    # 2-1. split train / test
    X = dataset.loc[:TRAIN_SHAPE - 1, [x for x in dataset.columns if x not in not_train and x not in need_encode]]
    y = dataset.loc[:TRAIN_SHAPE - 1, 'fraud_ind']
    print('\tTrain dataset shape :', X.shape)
    print('\tTrain label shape :', y.shape, '\n')

    # 2-2. get threhold
    th = Threshold()
    df = th.calc_threshold_diff(X, y, cat, n_fold=n_fold)
    best_threshold = th.get_best_threshold(df)
    print('\nBest Threshold = ', best_threshold)

    # 3. Training
    print('\n[Step 3/3] Start Training ... \n')
    if action == 'cv':
        cv = CrossValidate(threshold=best_threshold)
        res = cv.expanding_window(X, y, cat, boost_round=1000)
        print('>> Avg Cross Validation : {}'.format(sum(res) / len(res)))
        print('>> base line : 0.6034704709308101')
    elif action == 'submit':
        split, gain = train_submit(dataset, cat, not_train + need_encode, file_name=file_name)
        print('\nPrediction written to ./submit/{}.csv'.format(file_name))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--action", "-a", choices=['cv', 'submit'], default='cv', type=str)
    parser.add_argument("--feature", "-f", choices=['new', 'load'], default='new', type=str)
    parser.add_argument("--feature_fname", "-fn", default='feature_ver1', type=str)
    parser.add_argument("--output_fname", "-on", default='submit001', type=str)
    parser.add_argument("--n_fold", "-n", default=5, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    train(
        action=args.action,
        file_name=args.output_fname,
        feature=args.feature,
        feature_fname=args.feature_fname,
        n_fold=args.n_fold
    )