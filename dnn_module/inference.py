import gc
import os
import argparse

# import pickle as pkl
# import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from dataset import Features
from model import Net
from loss import FocalLoss


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def inference(model, testset_loader, threshold):
    print('Start Inferencing ...')
    os.makedirs('./submit/', exist_ok=True)
    softmax = nn.Softmax()
    model.eval()  # Important: set evaluation mode

    ids_all = None
    pure_output_all = None
    pred_max_all = None
    pred_threshold_all = None
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, (features, ids) in enumerate(testset_loader):
            print('\r[{}/{}]'.format(i, len(testset_loader)), end='')
            features, _ids = features.to(device), ids.squeeze(1).to(device)
            output = model(features)
            pred_max = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            pred_threshold = softmax(output).cpu() > threshold
            if ids_all is not None:
                ids_all = torch.cat([ids_all, ids], dim=0)
                pure_output_all = torch.cat([pure_output_all, output], dim=0)
                pred_max_all = torch.cat([pred_max_all, pred_max], dim=0)
                pred_threshold_all = torch.cat([pred_threshold_all, pred_threshold], dim=0)
            else:
                ids_all = ids
                pure_output_all = output
                pred_max_all = pred_max
                pred_threshold_all = pred_threshold

        ids_all = ids_all.detach().numpy()
        pure_output_all = pure_output_all.detach().numpy()
        pred_max_all = pred_max_all.detach().numpy()
        pred_threshold_all = pred_threshold_all.detach().numpy()
        pred_threshold_all = pred_threshold_all[:, 1]
        
        id_output = np.concatenate((ids_all, pure_output_all), axis=1)
        id_pred_max = np.concatenate((ids_all, pred_max_all), axis=1)
        id_pred_threshold = np.concatenate((ids_all, pred_threshold_all), axis=1)
        np.savetxt('./submit/id_pure_output.csv', id_output, delimiter=',', fmt='%0.4f', header='txkey,output')
        np.savetxt('./submit/id_pred_max.csv', id_pred_max, delimiter=',', fmt='%d', header='txkey,fraud_ind')
        np.savetxt('./submit/id_pred_softmax_threshold.csv', id_pred_threshold, delimiter=',', fmt='%d', header='txkey,fraud_ind')

def get_device():
    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    return device

def args_parse(a=0, g=1, t=1):
    parser = argparse.ArgumentParser()

    alpha_list = [0.25, 0.5, 0.75]          # 3
    gamma_list = [0.5, 1, 2, 5]             # 4
    threshold_list = [0.15, 0.248, 0.4]     # 3

    print('\n\n Focal_a{}_g{}_t{}'.format(str(a), str(g), str(t)))
    
    parser.add_argument('--test_size', type=int, default=1000, help='input test batch size') # 769
    parser.add_argument("--action", type=str, default='load', choices=['load', 'new'], help="action to load or generate new features")
    parser.add_argument("--model_name", type=str, default='Focal_a{}_g{}_t{}'.format(str(a), str(g), str(t)), help="model name for saving pth file")
    parser.add_argument('--alpha', type=float, default=alpha_list[a], help='alpha param of focal loss')
    parser.add_argument('--gamma', type=float, default=gamma_list[g], help='gamma param of focal loss')
    parser.add_argument('--threshold', type=float, default=threshold_list[t], help='alpha param of focal loss')

    opt = parser.parse_args()
    print(opt)
    return opt

if __name__ == '__main__':
    # parse training args
    opt = args_parse(a=0, g=0, t=0)
    model_path = os.path.join('.', 'models', '{}_final.pth'.format(opt.model_name))

    # get dataset 
    testset = Features(data_type='test', action=opt.action, feature_fname='FeatureOrigin')
    print('rows in testset:', len(testset)) # Should print 1217428

    # Use the torch dataloader to iterate through the dataset
    testset_loader = DataLoader(testset, batch_size=opt.test_size, shuffle=False)

    device = get_device()
    model = Net().to(device) # Remember to move the model to "device"
    load_checkpoint(model_path, model)
    print(model)

    inference(model, testset_loader, threshold=opt.threshold)
