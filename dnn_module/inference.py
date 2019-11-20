import gc
import os
import argparse

# import pickle as pkl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from dataset import Features
from model import Net, Net1D, Net1D_2, Net2D, Net2D_2
from loss import FocalLoss


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def inference(model, testset_loader, opt, threshold):
    if opt.infer_val:
        os.makedirs('./infer_val/', exist_ok=True)
    else:
        os.makedirs('./submit/', exist_ok=True)
    softmax = nn.Softmax()
    model.eval()  # Important: set evaluation mode

    ids_all = None
    output_softmax_all = None
    softmax_threshold_all = None
    labels_all = None
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, loader_iter in enumerate(testset_loader):
            print('\r[{}/{}]'.format(i, len(testset_loader)), end='')
            if opt.infer_val:
                features, ids, labels = loader_iter
            else:
                features, ids = loader_iter
            features = features.to(device)
            output = model(features)
            output_softmax = softmax(output).cpu()
            softmax_threshold = output_softmax > threshold
            
            if ids_all is not None:
                ids_all = torch.cat([ids_all, ids], dim=0)
                output_softmax_all = torch.cat([output_softmax_all, output_softmax], dim=0)
                if opt.infer_val:
                    labels_all = torch.cat([labels_all, labels], dim=0)
                else:
                    softmax_threshold_all = torch.cat([softmax_threshold_all, softmax_threshold], dim=0)
            else:
                ids_all = ids
                output_softmax_all = output_softmax
                if opt.infer_val:
                    labels_all = labels
                else:
                    softmax_threshold_all = softmax_threshold

        ids_all = ids_all.detach().cpu().numpy()
        output_softmax_all = output_softmax_all.detach().cpu().numpy()
        if opt.infer_val:
            labels_all = labels_all.detach().cpu().numpy()
            output_softmax_all = np.expand_dims(output_softmax_all[:, 1], axis=1)
            output_infer_val = np.concatenate((ids_all, output_softmax_all, labels_all), axis=1)
            df = pd.DataFrame(output_infer_val)
            df.columns = ['txkey', 'pred_softax', 'label']
            df.txkey = df.txkey.astype(int)
            df.label = df.label.astype(int)
            with open('./infer_val/InferVal_{}.pkl'.format(opt.model_name), 'wb') as file:
                pkl.dump(output_infer_val, file)
        else:
            softmax_threshold_all = softmax_threshold_all.detach().cpu().numpy()
            softmax_threshold_all = np.expand_dims(softmax_threshold_all[:, 1], axis=1)
        
            id_output_softmax = np.concatenate((ids_all, output_softmax_all), axis=1)
            np.savetxt('./submit/id_output_softmax.csv', id_output_softmax, delimiter=',', fmt='%0.4f', header='txkey,output')
            id_softmax_threshold = np.concatenate((ids_all, softmax_threshold_all), axis=1)
            np.savetxt('./submit/id_softmax_threshold.csv', id_softmax_threshold, delimiter=',', fmt='%d', header='txkey,fraud_ind')

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
    
    parser.add_argument('--infer_val', action="store_true", help='inference not on testset but on valset')
    parser.add_argument("--model_dim", type=str, default='1D', choices=['old','1D', '2D'], help="model choice")
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
    testset = Features(data_type='infer', dim=opt.model_dim, action=opt.action, feature_fname='FeatureOrigin', dump_id=opt.infer_val)
    print('rows in testset:', len(testset)) # Should print 1217428

    # Use the torch dataloader to iterate through the dataset
    testset_loader = DataLoader(testset, batch_size=opt.test_size, shuffle=False)

    device = get_device()
    if opt.model_dim == '1D':
        model = Net1D_2().to(device)
    elif opt.model_dim == '2D':
        model = Net2D_2().to(device)
    elif opt.model_dim == 'old':
        model = Net().to(device)
    load_checkpoint(model_path, model)
    print(model)

    if opt.infer_val:
        print('\n\nStart inferencing on valset to dump pkl of DataFrame\n\n')
    else:
        print('\n\nStart inferencing on testset\n\n')
    
    inference(model, testset_loader, opt, threshold=opt.threshold)
