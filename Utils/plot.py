import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
font = {'weight' : 'bold',
        'size'   : 22}
plt.rc('font', **font)
import seaborn as sns

def plot_high_fraud(combine, col, ax=None):
    df1 = combine[[col, 'fraud_ind']].groupby(col).mean().reset_index()
    ax.scatter(x='fraud_ind', y=col, data=df1, c='C1')
    if(ax):
        ax.set_title(f'fraud rate for individual {col}')
    else:
        ax.title(f'fraud rate for individual {col}')
    del df1
    gc.collect()
    
def plot_high_count(combine, col, ax=None):
    df1 = combine[col].value_counts().to_frame().reset_index()
    ax.scatter(x=col, y='index', data=df1)
    
    if(ax):
        ax.set_xlabel('counts')
        ax.set_ylabel(col)
        ax.set_title(f'value counts for {col}')
    else:
        ax.xlabel('counts')
        ax.ylabel(col)
        ax.title(f'value counts for {col}')
    del df1
    gc.collect()
    
def plot_high_countfraud(combine, col, ax=None):
    df1 = combine[col].value_counts().to_frame().reset_index().rename(columns={'index': col, col: 'counts'})
    df2 = combine[[col, 'fraud_ind']].groupby(col).mean().reset_index()
    df3 = pd.merge(df1, df2, on=col)
    ax.scatter(x='counts', y='fraud_ind', data=df3, c='C2')
    
    if(ax):
        ax.set_title(f'fraud rate respect to counts for {col}')
    else:
        ax.title(f'fraud rate respect to counts for {col}')
    del df1, df2, df3
    gc.collect()
    
def plot_low_fraud(combine, col, ax=None):
    df1 = combine[[col, 'fraud_ind']].groupby(col).mean().reset_index()
    if(ax):
        sns.barplot(x=col, y='fraud_ind', data=df1, ax=ax)
    else:
        sns.barplot(x=col, y='fraud_ind', data=df1)
    del df1
    gc.collect()

def plot_low_count(combine, col, ax=None):
    if(ax):
        sns.countplot(combine[col], ax=ax)
    else:
        sns.barplot(combine[col])
        
def plot_low_countfraud(combine, col, ax=None):
    df1 = combine[col].value_counts().to_frame().reset_index().rename(columns={'index': col, col: 'counts'})
    df2 = combine[[col, 'fraud_ind']].groupby(col).mean().reset_index()
    df3 = pd.merge(df1, df2, on=col)
    if(ax):
        sns.barplot(x='counts', y='fraud_ind', data=df3, ax=ax)
    else:
        sns.barplot(x='counts', y='fraud_ind', data=df3)
    del df1, df2, df3
    gc.collect()

def plot_dist_diff(combine, col, ax=None):
    if(ax):
        sns.distplot(combine.loc[(combine['fraud_ind'] == 1), col], ax=ax)
        sns.distplot(combine.loc[(combine['fraud_ind'] == 0), col], ax=ax)
    else:
        sns.distplot(combine.loc[(combine['fraud_ind'] == 1), col])
        sns.distplot(combine.loc[(combine['fraud_ind'] == 0), col])
        