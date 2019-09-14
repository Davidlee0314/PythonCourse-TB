import pandas as pd
import numpy as np 

def map_stat_feature(X, b, c, mean=True, max=True, \
        min=True, std=True, var=True, median=True, median_diff=True):
    # b groupby c mean...
    X[b + '_' + c + '_mean'] = \
        X[c].map(X.loc[:, [b, c]].groupby(c).mean().loc[:, b].to_dict())
    X[b + '_' + c + '_max'] = \
        X[c].map(X.loc[:, [b, c]].groupby(c).max().loc[:, b].to_dict())
    X[b + '_' + c + '_min'] = \
        X[c].map(X.loc[:, [b, c]].groupby(c).mean().loc[:, b].to_dict())
    X[b + '_' + c + '_std'] = \
        X[c].map(X.loc[:, [b, c]].groupby(c).std().loc[:, b].to_dict())
    X[b + '_' + c + '_var'] = X[b + '_' + c + '_std'] ** 2
    X[b + '_' + c + '_median'] = \
        X[c].map(X.loc[:, [b, c]].groupby(c).median().loc[:, b].to_dict())
    X[b + '_' + c + '_median_diff'] = \
        X[b] - X[b + '_' + c + '_median']
    return None

class FeatureEngineer():
    def __init__(self, ):
        return None
    
    # 1.
    def numerical_stat(self, df, num_col, columns, \
        mean=True, max=True, min=True, std=True, var=True, median=True, median_diff=True):
        '''
        Mapping numerical feature to categorical feature, and get statistical value

        df: the dataframe you want to apply
        num_col: the numerical columns need to be groupby
        columns: list of columns want to interact with money
        mean: need mean inforamtion
        max: need max information
        std: need std information
        var: need var information
        median: need median information
        median_diff: need median differenc information
        '''
        if(not isinstance(columns, list)):
            raise ValueError('columns params need to be a list')
        for col in columns:
            map_stat_feature(df, num_col, col, mean=mean, max=max, min=min, std=std, \
                var=var, median=median, median_diff=median_diff)  
        return None
    
    # 2. 
    def card_num(self, df):
        df['card_num'] = df['bank'].map(df.loc[:, ['card', 'bank']].groupby('bank').count().loc[:,'card'].to_dict())
        return None

    def engineer_all(self, df):
        # 1.
        high_1 = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation']
        # self.numerical_stat(df, 'money', high_1)
        # self.numerical_stat(df, 'iterm', high_1)

        # 2.
        # card_num: one bank has how many cards. ( 一個帳號有幾張卡 )
        self.card_num(df)

        # 3.

        return None