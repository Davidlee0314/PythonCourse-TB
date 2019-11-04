import gc
import os
import sys
sys.path.append('..')

import pickle as pkl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from Utils.Feature import FeatureEngineer
from Utils import read_data

class Features(Dataset):
    def __init__(self, data_type='train', action='new', feature_fname='FeatureOrigin'):
        """ Intialize the dataset """
        self.TRAIN_SHAPE = 1521787
        self.VAL_SHAPE = int(1521787 * 0.8)
        self.not_train = ['txkey', 'date', 'time']  # fraud_ind
        self.need_encode = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation']
        self.category_col_list = ['status', 'trade_cat', 'pay_type', 'trade_type', '3ds', 'fallback', 'hour']
        self.feature_root = './features/'
        os.makedirs(self.feature_root, exist_ok=True)

        self.dataset = self.get_engineered_data(data_type, action, feature_fname)
        self.len = self.dataset.shape[0]
        
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        row_feature = self.dataset.iloc[index]

        label = int(row_feature['fraud_ind'])
        row_feature = row_feature.drop(labels=["fraud_ind"])
        label = torch.tensor([label])
        row_feature = torch.tensor(list(row_feature))
        # print(row_feature.shape)
        # print('label :', label)
        # print('type(label) :', type(label))

        return row_feature, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

    def get_engineered_data(self, data_type, action, feature_fname):
        feature_path = os.path.join(self.feature_root, feature_fname+'.pkl')

        ###
        if os.path.exists(feature_path):
            print("action = 'load'")
            action = 'load'
        else:
            print("action = 'new'")
            action = 'new'
        ###

        if action == 'load':
            with open(feature_path, 'rb') as file:
                combine = pkl.load(file)
            print('Features loaded (from {})'.format(feature_path))
        elif action == 'new':
            # get basic train test data
            train = read_data.read('../data/train_mr.pkl')
            test_ = read_data.read('../data/test_mr.pkl')
            combine = pd.concat([train, test_])
            combine = combine.reset_index(drop=True)

            # add hour feature
            print('initial dataset.shape =', combine.shape)
            fe = FeatureEngineer()
            fe.get_hour(combine)
            print('add hour dataset.shape =', combine.shape)

            # get pre-processed feature data
            prefix = '../data/mine/'
            file_names = ['need_encode_money_stat.pkl', 
                'need_encode_oneday_count.pkl', 
                'need_encode_sum_money.pkl', 
                'tradenum_bank-col.pkl', 
                'tradenum_col-col.pkl', 
                'tradenum_bank-col-col.pkl',    # 'group_col.pkl' , 'again&group.pkl'
            ]
            files_list = [prefix + x for x in file_names]
            for file_name in files_list:
                with open(file_name, 'rb') as f:
                    temp = pkl.load(f)
                    combine = pd.concat([combine, temp], axis=1)
                    print(file_name + ' loaded, shape =', combine.shape)
                    del temp
                    gc.collect()
            
            # fill NaN / catgory column >> get dummy
            combine['3ds'].fillna(value=2, inplace=True)
            combine['fallback'].fillna(value=2, inplace=True)

            combine = pd.get_dummies(combine, columns=self.category_col_list)
            combine.fillna(value=-1, inplace=True)
            print('dataset.shape (after get_dummies / fillna):', combine.shape)

            # remove test set (no label)
            combine = combine.loc[:self.TRAIN_SHAPE - 1, [x for x in combine.columns if x not in self.not_train]]

            # save features
            with open(feature_path, 'wb') as file:
                pkl.dump(combine, file)

        # split train_set / val_set
        val_set = combine.iloc[self.VAL_SHAPE:]
        train_set = combine.iloc[:self.VAL_SHAPE - 1]
        print('dataset.shape (after get_dummies / ignore not_train):', combine.shape)
        print('train_set shape = ', train_set.shape)
        print('val_set shape = ', val_set.shape, '\n')
        del combine
        gc.collect()

        if data_type == 'train':
            del val_set
            gc.collect()
            return train_set
        elif data_type == 'val':
            del train_set
            gc.collect()
            return val_set

if __name__ == '__main__':
    trainset = Features(data_type='train', action='load', feature_fname='FeatureOrigin')
    print('rows in trainset:', len(trainset)) # Should print 60000

    # Use the torch dataloader to iterate through the dataset
    trainset_loader = DataLoader(trainset, batch_size=4, shuffle=True)

    # get some random training samples
    dataiter = iter(trainset_loader)
    features, labels = dataiter.next()
    print('Feature tensor in each batch:', features.shape, features.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype)