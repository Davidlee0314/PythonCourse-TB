#%%
import pandas as pd
import numpy as np
from Utils.CrossValidate import CrossValidate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb
import gc
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

#%%
train = pd.read_csv('data/train_sort.csv')
test = pd.read_csv('data/test_sort.csv')

#%%
X = train[[x for x in train.columns if x != 'txkey' and x != 'locdt' and x != 'loctm']]
for c in X.columns:
    if(X[c].dtype == object):
        li = list(X[c].unique())
        li  = [x for x in li if str(x) != 'nan']
        dic = {}
        for i, item in enumerate(li):
            dic[item] = i
        X[c] = X[c].map(dic)
        del li, dic
X.info()
#%%
y = X['fraud_ind']
X.drop(['fraud_ind'], axis=1, inplace=True)

#%%
params = {
            'objective': 'binary',
            'early_stopping_rounds': 100,
            'learning_rate': 0.01,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'max_depth': -1,
            'num_leaves': 100,
            'seed': 6
        }
# cv_test = pd.DataFrame(
#     columns=['feature_num', 'slide', 'expand', 'expand_last2', 'ratio', 'ratio_last2']
# )
# cv_test

#%%
X.shape

#%%
import warnings
warnings.filterwarnings('ignore')

cv = CrossValidate()
for i in [3]:
    cat = [x for x in X.iloc[:, :i].columns if x != 'conam' and x != 'iterm']
    slide = cv.sliding_window(X.iloc[:, :i], y, cat, boost_round=500)
    ex = cv.expanding_window(X.iloc[:, :i], y, cat, boost_round=500)
    ex_2 = cv.expanding_window(X.iloc[:, :i], y, cat, boost_round=500, last_fold=2)
    ratio = cv.ratio_window(X.iloc[:, :i], y, cat, boost_round=500)
    ratio_2 = cv.ratio_window(X.iloc[:, :i], y, cat, boost_round=500, last_fold=2)
    cv_temp = pd.DataFrame({
        'feature_num': [i], 
        'slide': [slide], 
        'expand': [ex], 
        'expand_last2': [ex_2], 
        'ratio': [ratio], 
        'ratio_last2': [ratio_2],
        'public': [0]
    })
    cv_test = cv_test.append(cv_temp)

#%%
cv_test.head()

#%%
X_te = test[[x for x in test.columns if x != 'txkey' and x != 'locdt' and x != 'loctm']]
for c in X_te.columns:
    if(X_te[c].dtype == object):
        li = list(X_te[c].unique())
        li  = [x for x in li if str(x) != 'nan']
        dic = {}
        for i, item in enumerate(li):
            dic[item] = i
        X_te[c] = X_te[c].map(dic)
        del li, dic
# X.info()

#%%
for i in [3]:
    cat = [x for x in X.iloc[:, :i].columns if x != 'conam' and x != 'iterm']
    train_data = lgb.Dataset(data=X.iloc[:, :i], label=y, categorical_feature=cat)
    clf = lgb.train(params, 
        train_data,
        valid_sets=[train_data],
        num_boost_round=500,
        verbose_eval=50, 
        feval=lgb_f1_score)
    prob = clf.predict(X_te.iloc[:, :i])
    pred = np.where(prob > 0.5, 1, 0)
    submit = pd.DataFrame({
        'txkey': test['txkey'],
        'fraud_ind': pred
    })
    submit.to_csv('data/submit/cv_test_' + str(i) + '.csv', index=False)
    del train_data, submit
    gc.collect()
    

#%%
cv_test.loc[cv_test['feature_num'] == 3, 'public'] = 0.351814

#%%
# cv_test = pd.DataFrame({
#     'feature_num': [13, 15, 17, 19],
#     'slide': [0.755764, 0.797424, 0.800481, 0.822625],
#     'expand': [0.483414, 0.498328, 0.497661, 0.507765],
#     'expand_last2': [0.470156, 0.473215, 0.469880, 0.483494],
#     'ratio': [0.504114, 0.529631, 0.528722, 0.542652],
#     'ratio_last2': [0.471640, 0.483171, 0.479555, 0.493391],
#     'public': [0.475711, 0.500132, 0.508891, 0.525984]
# })
cv_test.sort_values(by='feature_num')
#%%
import seaborn as sns 
sns.heatmap(cv_test.corr(), annot=True)


#%%
cv_test.to_csv('data/cv_test.csv', index=False)

#%%
