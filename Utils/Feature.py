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

def trade_ratio(X, b, c):
    # b grouped by c
    dct = (X.groupby([c, b]).size() / X.groupby(c).size()).to_dict()
    X[b + '_' + c + '_traderatio']  = X.set_index([c, b]).index.map(dct.get)
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
        '''
        one bank has how many cards. ( 一個帳號有幾張卡 )
        '''
        df['card_num'] = df['bank'].map(df.loc[:, ['card', 'bank']].groupby('bank').count().loc[:,'card'].to_dict())
        return None

    # 4.
    def coin_country_dominance(self, df):
        '''
        dominant coin type 貨幣在那個國家交易過幾次
        '''
        dct = (df.groupby(['nation','coin']).size() / df.groupby('nation').size()).to_dict()
        df['coin_country_dominance'] = df.set_index(['nation', 'coin']).index.map(dct.get)
        return None

    # 9.
    def money_divide_term(self, df):
        '''
        money / term => term_money 分期金額
        '''
        df['money_term'] = df['money'] / (df['term'] + 1)
        return None

    # 11.
    def count_city_num(self, df):
        '''
        一個國家有幾個城市
        '''
        df['city_num'] = df['nation'].map(df.loc[:, ['city', 'nation']].groupby('nation').count().loc[:,'city'].to_dict())
        return None

    # 12.
    def categorical_dummy_ratio(self, df, cats, dummies):
        '''
        dummy variable for each categorical feature trade ratio
        ex. nation / acquirer / bank, fallback / online / 3ds / install / excess 的交易數量比例
        
        cats: list for categorical features
        dummies: list for dummy features
        '''
        if(not isinstance(cats, list)):
            raise ValueError('cats params need to be a list')
        if(not isinstance(dummies, list)):
            raise ValueError('dummies params need to be a list')
        for c in cats:
            for b in dummies:
                trade_ratio(df, b, c)
        return None

    # 15.
    def week_day(self, df):
        '''
        week day of the trade
        '''
        df['week_day'] = df['date'] % 7
        return None


    def engineer_all(self, df):
        # 1.
        high = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation']
        # self.numerical_stat(df, 'money', high)
        # self.numerical_stat(df, 'iterm', high)

        # 2.
        self.card_num(df)

        # 4.
        self.coin_country_dominance(df)

        # 9.
        self.money_divide_term(df)

        # 11.
        self.count_city_num(df)

        # 12.
        binary_category = ['fallback', 'online', '3ds', 'install', 'excess']
        group_category = ['nation', 'acquirer', 'bank']
        self.categorical_dummy_ratio(df, group_category, binary_category)

        # 15.
        self.week_day(df)
        return None