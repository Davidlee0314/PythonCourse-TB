import os
import numpy as np
import pandas as pd
import gc

# model
from sklearn import preprocessing, model_selection, metrics
from Utils.CrossValidate import CrossValidate, lgb_train
import lightgbm as lgb
TRAIN_SHAPE = 1521787
ROUTE = '/Users/davidlee/python/TBrain/data/'

# covariate shift
def covariate_shift(combine, feature, train_shape):
    df_train = pd.DataFrame(data={feature: combine.loc[:train_shape, feature], 'isTest': 0})
    df_test = pd.DataFrame(data={feature: combine.loc[train_shape:, feature], 'isTest': 1})
    
    # Creating a single dataframe
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Encoding if feature is categorical
    if str(df[feature].dtype) in ['object', 'category']:
        df[feature] = preprocessing.LabelEncoder().fit_transform(df[feature].astype(str))
    
    # Splitting it to a training and testing set
    X_train, X_test, y_train, y_test = model_selection.train_test_split(df[feature], df['isTest'], test_size=0.33, random_state=47, stratify=df['isTest'])
    params = {
        'objective': 'binary', 
        "boosting_type": "gbdt", 
        "subsample": 1, 
        "bagging_seed": 11, 
        "metric": 'auc'
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train.values.reshape(-1, 1), y_train)
    roc_auc =  metrics.roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1])
    if(roc_auc < 0.5):
        roc_auc = 1 - roc_auc
    del df, X_train, y_train, X_test, y_test
    gc.collect()
    print(feature, 'roc_auc score equals', roc_auc)
    return None

def get_unsample_data(df, n):
    '''
    df: training dataframe
    n: the ratio of the normal and anamoly
    '''
    number_records_fraud = len(df[df['fraud_ind'] == 1])
    fraud_indices = np.array(df[df['fraud_ind'] == 1].index)
    normal_indices = df[df['fraud_ind'] == 0].index

    random_normal_indices = np.random.choice(normal_indices, number_records_fraud * n, replace=False)
    random_normal_indices = np.array(random_normal_indices)

    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    under_sample_data = df.iloc[under_sample_indices, :].copy().sample(frac=1).reset_index(drop=True)
    return under_sample_data

def fast_validate(combine, cat, not_train):
    cv = CrossValidate()
    res_list = []
    for i in [2, 3, 4]:
        print('*' * 10, i, 'splits', '*' * 10)
        temp = combine.loc[(combine['date'] > 0) & (combine['date'] <= 10), :].copy()
        X = temp.loc[:, [x for x in temp.columns if x not in not_train]]
        y = temp.loc[:, 'fraud_ind']
        print(X.shape)
        res = cv.expanding_window(X, y, cat, boost_round=1000, n_fold=i)
        res_list.append(sum(res) / len(res))
        del temp, X, y, res
        gc.collect()
    return sum(res_list) / len(res_list)

def train_submit(combine, cat, not_train, threshold, file_name, boost_round=1000):
    X = combine.loc[:TRAIN_SHAPE - 1, [x for x in combine.columns if x not in not_train]]
    y = combine.loc[:TRAIN_SHAPE - 1, 'fraud_ind']
    print('X.shape :',X.shape)
    train_data = lgb.Dataset(data=X, label=y, categorical_feature=cat) 
    val_data = None

    # model training
    clf = lgb_train(train_data, val_data, threshold, boost_round=boost_round, for_submit=True)

    # predicting
    test = combine.loc[TRAIN_SHAPE:, [x for x in combine.columns if x not in not_train]]
    pred = clf.predict(test)
    submit = pd.DataFrame({
        'txkey': combine.loc[TRAIN_SHAPE:, 'txkey'],
        'fraud_ind': np.where(pred >= threshold, 1, 0)
    })
    os.makedirs('./submit', exist_ok=True)
    submit.to_csv(f'./submit/{file_name}.csv', index=False)
    # submit.to_csv(ROUTE + f'submit/{ file_name }.csv', index=False)
    del X, y, train_data, test, submit
    gc.collect()
    return clf.feature_importance(importance_type='split'), clf.feature_importance(importance_type='gain')


