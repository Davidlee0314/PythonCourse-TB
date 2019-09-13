import numpy as np
import pandas as pd
import gc

# model
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb

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