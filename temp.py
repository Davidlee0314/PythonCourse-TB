#%%
%load_ext autoreload
%autoreload 2
#%%
import pandas as pd
import numpy as np
import Utils
from Utils.CrossValidate import CrossValidate, lgb_f1_score
import lightgbm as lgb
import gc
TRAIN_SHAPE = 1521787
#%%
combine = pd.read_csv('data/combine.gz', compression='gzip')
#%%
combine.info()
#%%
not_train = ['txkey', 'locdt', 'loctm', 'minute', 'second', 'absolute_time', 'fraud_ind']
X = combine.loc[:TRAIN_SHAPE - 1, [x for x in combine.columns if x not in not_train]]
cat = [x for x in X if x != 'conam' and x != 'iterm']
cat
#%%
y = combine.loc[:TRAIN_SHAPE - 1, 'fraud_ind']
y
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
train_data = lgb.Dataset(data=X, label=y, categorical_feature=cat)

#%%
clf = lgb.train(params, 
    train_data,
    valid_sets=[train_data],
    num_boost_round=1000,
    verbose_eval=50, 
    feval=lgb_f1_score)
#%%
test = combine.loc[TRAIN_SHAPE:, [x for x in combine.columns if x not in not_train]]
pred = clf.predict(test)

#%%
submit = pd.DataFrame({
    'txkey': combine.loc[TRAIN_SHAPE:, 'txkey'],
    'fraud_ind': np.where(pred > 0.5, 1, 0)
})
submit.to_csv('data/submit/add_hour.csv')
#%%
cv.add_cv_public(0.539367)
