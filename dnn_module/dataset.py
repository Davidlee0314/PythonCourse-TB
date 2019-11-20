import gc
import os
import sys
sys.path.append('..')

import pickle as pkl
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Utils.Feature import FeatureEngineer
from Utils import read_data

class Features(Dataset):
    def __init__(self, dim='1D', data_type='train', model_type='cnn', action='new', feature_fname='FeatureOrigin', infer_val=False):
        """ Intialize the dataset """
        self.TRAIN_SHAPE = 1521787
        self.VAL_SHAPE = int(1521787 * 0.8)
        self.not_train = ['txkey', 'date', 'time']  # fraud_ind
        self.not_test = ['date', 'time', 'fraud_ind']
        self.need_encode = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation']
        self.category_col_list = ['status', 'trade_cat', 'pay_type', 'trade_type', '3ds', 'fallback', 'hour']
        self.feature_root = './features/'
        os.makedirs(self.feature_root, exist_ok=True)

        self.data_type = data_type      # train / val / infer / full_train
        self.model_type = model_type
        self.dim = dim  # 1D / 2D / old
        self.infer_val = infer_val          # val set, but need id to check
        self.dataset = self.get_engineered_data(data_type, action, feature_fname)
        if model_type == 'cnn':
            self.len = self.dataset.shape[0]    # DataFrame
        elif model_type == 'rnn':
            self.len = len(self.dataset)        # list of DataFrames
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        if self.model_type == 'cnn':
            row_feature = self.dataset.iloc[index]
            if self.data_type == 'infer':
                key = 'txkey'
            elif self.data_type in ['train', 'val', 'full_train']:
                key = 'fraud_ind'
            label_or_id = int(row_feature[key])
            row_feature = row_feature.drop(labels=[key])
            label_or_id = torch.tensor([label_or_id])
            row_feature = torch.tensor(list(row_feature))
            if self.dim == '2D':
                zero = torch.zeros(42)
                row_feature = torch.cat([row_feature , zero])   # 534 + 42 = (576)
                row_feature = row_feature.view(24, 24)          # (576) >> (24, 24)
                row_feature = row_feature.unsqueeze(0)          # (24, 24) >> (1, 24, 24)
            if self.infer_val:    # val set, but need id to check
                id = int(row_feature['txkey'])
                label = label_or_id
                return row_feature, id, label
            return row_feature, label_or_id
        elif self.model_type == 'rnn':
            # TODO : design for LSTM input
            pass

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

    def get_engineered_data(self, data_type, action, feature_fname):
        feature_path = os.path.join(self.feature_root, feature_fname+'.pkl')
        test_feature_path = os.path.join(self.feature_root, feature_fname+'_test.pkl')
        if action == 'load':
            if data_type == 'infer':
                if self.infer_val:
                    pass
                else:
                    if not os.path.exists(test_feature_path):
                        raise FileNotFoundError('{} not exists, please select another file.'.format(test_feature_path))
                    with open(test_feature_path, 'rb') as file:
                        testset = pkl.load(file)
                    print('Test Features loaded (from {})'.format(test_feature_path))
            elif data_type in ['train', 'val', 'full_train'] or self.infer_val:
                if not os.path.exists(feature_path):
                    raise FileNotFoundError('{} not exists, please select another file.'.format(feature_path))
                with open(feature_path, 'rb') as file:
                    train_val = pkl.load(file)
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
                'tradenum_bank-col-col.pkl',
                'embed_money.pkl',
                'embed_online.pkl',
                'embed_tradetype.pkl',
                'graph_embed_w3.pkl'
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

            # remove some columns
            testset = combine.loc[self.TRAIN_SHAPE:, [x for x in combine.columns if x not in self.not_test]]
            train_val = combine.loc[:self.TRAIN_SHAPE - 1, [x for x in combine.columns if x not in self.not_train]]
            del combine
            gc.collect()
            # save features
            with open(test_feature_path, 'wb') as file:
                pkl.dump(testset, file)
            with open(feature_path, 'wb') as file:
                pkl.dump(train_val, file)
            
        # train / val / infer
        if data_type == 'infer':
            if self.infer_val:
                val_set = train_val.iloc[self.VAL_SHAPE:]
                print('val_set(for inference)  shape = ', val_set.shape, '\n')
                del train_val
                gc.collect()
                return val_set
            else:
                print('testset.shape (after get_dummies / ignore not_train):', testset.shape)
                return testset
        elif data_type == 'full_train':
            print('dataset.shape (after get_dummies / ignore not_train):', train_val.shape)
            return train_val
        elif data_type in ['train', 'val']:
            # split train_set / val_set
            print('dataset.shape (after get_dummies / ignore not_train):', train_val.shape)
            if data_type == 'train':
                train_set = train_val.iloc[:self.VAL_SHAPE - 1]
                print('train_set shape = ', train_set.shape)
                output_set = train_set
            elif data_type == 'val':
                val_set = train_val.iloc[self.VAL_SHAPE:]
                print('val_set shape = ', val_set.shape, '\n')
                output_set = val_set
            del train_val
            gc.collect()

            if self.model_type == 'cnn':
                return output_set    # DataFrame
            elif self.model_type == 'rnn':
                SubDFList = []
                for item in output_set['bank'].unique():
                    sub_df = output_set[output_set['bank'] == item]
                    SubDFList.append(sub_df)
                return SubDFList    # list of DataFrames

if __name__ == '__main__':
    # trainset = Features(data_type='train', action='load', feature_fname='FeatureOrigin')
    # print('rows in trainset:', len(trainset)) # Should print 60000

    # # Use the torch dataloader to iterate through the dataset
    # trainset_loader = DataLoader(trainset, batch_size=4, shuffle=True)

    # # get some random training samples
    # dataiter = iter(trainset_loader)
    # features, labels = dataiter.next()
    # print('Feature tensor in each batch:', features.shape, features.dtype)
    # print('Label tensor in each batch:', labels.shape, labels.dtype)




    ### testing
    
    sm = torch.Tensor(8, 2).uniform_(0, 1)
    print(sm.shape)
    sm = sm[:, 1]
    print(sm.shape)
    sm = np.expand_dims(sm.numpy(), axis=1)
    print(sm.shape, '\n============================softmax shape over')


    label = torch.Tensor(8, 1).uniform_(0, 1)
    id = torch.Tensor(8, 1).uniform_(0, 100)

    print(id.shape)
    print(label.shape)

    # out = torch.stack([id.squeeze(), sm, label.squeeze()], dim=1)ㄥ
    output_infer_val = np.concatenate((id, sm, label), axis=1)
    print(output_infer_val.shape)
    print(output_infer_val)
    df = pd.DataFrame(output_infer_val)
    df.columns = ['txkey', 'pred_softax', 'label']
    df.txkey = df.txkey.astype(int)
    df.label = df.label.astype(int)

    val_softmax_path = './test.pkl'


    with open(val_softmax_path, 'wb') as file:
        pkl.dump(df, file)


    print(df.shape)
    print(df)
    # print(out.shape)
    # df = pd.DataFrame(out.numpy())
    # df.columns = ['txkey', 'pred_softax', 'label']
    # print(df.shape)
    # print(df)


