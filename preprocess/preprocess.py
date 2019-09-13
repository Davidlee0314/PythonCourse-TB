#%%[markdown]
# feature engineering: 
# 1. value count
# 2. stat encoding / time window (mean, median, max, min, std)
# 3. diff of mean with adjeacent window
# 4. last appearance 
# 5. holiday
# 6. key
# 7. card_num
# 8. 

#%%[markdown]
# Import and Combine

#%%
import numpy as np
import pandas as pd 
pd.set_option('display.max_columns', None)
import gc
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import preprocessing, metrics, model_selection
import lightgbm as lgb 

#%%
train = pd.read_csv('/Users/davidlee/python/TBrain/data/sort/train_sort.csv')
test = pd.read_csv('/Users/davidlee/python/TBrain/data/sort/test_sort.csv')

#%%
cv_test= pd.read_csv('/Users/davidlee/python/TBrain/data/cv_test.csv')
cv_test

#%%
combine = pd.concat([train, test], ignore_index=True)
combine.shape

#%%
combine.isnull().sum()

#%%[markdown]
# Utils
def transform_loctm(combine):
    combine['loctm'] = combine['loctm'].astype(int).astype(str).str.zfill(6)
    combine['hour'] = combine['loctm'].str.slice(0, 2).astype(int)
    combine['minute'] = combine['loctm'].str.slice(2, 4).astype(int)
    combine['second'] = combine['loctm'].str.slice(4, 6).astype(int)
    combine['loctm'] = combine['loctm'].astype('int64')

def covariate_shift(combine, feature):
    df_train = pd.DataFrame(data={feature: combine.loc[:train.shape[0], feature], 'isTest': 0})
    df_test = pd.DataFrame(data={feature: combine.loc[train.shape[0]:, feature], 'isTest': 1})
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

def plot_categorical(combine, feature):
    (fig, axes)= plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    sns.barplot(combine[feature], combine['fraud_ind'], ax=axes[0])
    axes[0].set_title('Relations with fraud')
    sns.countplot(x=feature, data=combine, ax=axes[1])
    axes[1].set_title('Combine Countplot')
    sns.countplot(x=feature, data=combine.iloc[:train.shape[0]], ax=axes[2])
    axes[2].set_title('Tr Countplot')
    sns.countplot(x=feature, data=combine.iloc[train.shape[0]:], ax=axes[3])
    axes[3].set_title('Te Countplot')

def reduce_memory(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    for col in df.columns:
        print("******************************")
        print("Column: ", col)
        print("dtype before: ", df[col].dtype)
        if(df[col].dtype == object and col != 'absolute_time'):
            df[col] = df[col].map({'N': 0, 'Y': 1})
            if(df[col].isnull().sum() == 0):
                df[col] = df[col].astype(np.uint8)
            else:
                df[col] = df[col].astype(np.float32)
        elif(df[col].dtype == np.int64):
            mx = df[col].max()
            mn = df[col].min()
            if mn >= 0:
                if mx < 255:
                    df[col] = df[col].astype(np.uint8)
                elif mx < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif mx < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)    
        elif(df[col].dtype == np.float64):
            df[col] = df[col].astype(np.float32)
        print("dtype after: ",df[col].dtype)
        print("******************************")
    
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ", mem_usg ," MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")


#%%[markdown]
# Target variable

#%%
transform_loctm(combine)
combine.head()
#%%
import datetime 
startdate = datetime.datetime.strptime('2018-01-01', '%Y-%m-%d')
combine['absolute_time'] = startdate + pd.to_timedelta(combine['locdt'], 'days') + pd.to_timedelta(combine['hour'], 'hours') + \
                            pd.to_timedelta(combine['second'], 'seconds') + pd.to_timedelta(combine['minute'], 'minutes')
combine['absolute_time']
#%%
combine[['fraud_ind', 'absolute_time']].set_index('absolute_time').resample('0.5D').mean().plot()

#%%
plt.figure(figsize=(10, 7))
combine['absolute_time'].dt.floor('d').value_counts().sort_index().plot()

#%%[markdown]
# covariate check

for c in [x for x in combine.columns if x != 'txkey' and x != 'locdt' and x != 'loctm'\
            and x != 'absolute_time' and x != 'hour' and x != 'minute' and x != 'second' and x != 'fraud_ind']:
    print(c)
    covariate_shift(combine, c)

#%%
reduce_memory(test)

#%%
import pickle as pkl 
import pandas as pd 
combine=pd.read_csv('data/combine.gz', compression='gzip')

#%%


#%%
