import gc
import os
import argparse

# import pickle as pkl
# import pandas as pd
# import numpy as np
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


def save_checkpoint(checkpoint_path, model, optimizer):
    save_dir, _ = os.path.split(checkpoint_path)
    os.makedirs(save_dir, exist_ok=True)
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def eval(model, testset_loader, criterion, threshold):
    print('Start Evaluating ...')
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    labels_all = None
    output_all = None
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, (features, labels) in enumerate(testset_loader):
            print('\r[{}/{}]'.format(i, len(testset_loader)), end='')
            features, _labels = features.to(device), labels.squeeze(1).to(device)
            output = model(features)
            test_loss += criterion(output, _labels).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(_labels.view_as(pred)).sum().item()
            if labels_all is not None:
                labels_all = torch.cat([labels_all, labels], dim=0)
                output_all = torch.cat([output_all, output], dim=0)
            else:
                labels_all = labels
                output_all = output

    rows = labels_all.shape[0]
    labels_onehot = torch.zeros(rows, 2).scatter_(1, labels_all, 1)
    # 考慮類別的不平衡性，需要計算類別的加權平均 , average='weighted', 'macro'
    softmax = nn.Softmax()
    f1 = f1_score(labels_onehot.cpu(), softmax(output_all).cpu() > threshold, average='weighted')

    test_loss /= len(testset_loader.dataset)
    print('\nEval set: \n\tAverage loss: {:.4f} \n\tAccuracy: {:.0f}% ({}/{}) \n\tF1 Score: {}\n'.format(
        test_loss, 100. * correct / len(testset_loader.dataset), correct, len(testset_loader.dataset), f1))

def train_save(model, trainset_loader, testset_loader, opt, epoch=5, save_interval=4000, log_interval=1000, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    criterion = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
    # criterion = nn.CrossEntropyLoss()

    iteration = 0
    for ep in range(epoch):
        model.train()  # set training mode
        for batch_idx, (features, labels) in enumerate(trainset_loader):
            features, labels = features.to(device), labels.squeeze(1).to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(features), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('./models/{}_backup.pth'.format(opt.model_name), model, optimizer)
            iteration += 1
        # save_checkpoint('./models/{}_epoch_{}.pth'.format(opt.model_name, epoch), model, optimizer)
        eval(model, testset_loader, criterion, threshold=opt.threshold)
    
    # save the final model
    save_checkpoint('./models/{}_final.pth'.format(opt.model_name), model, optimizer)

def get_device():
    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    return device

def args_parse(a=0, g=0, t=0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_type", type=str, default='train', choices=['train', 'tune'], help="action to load or generate new features")

    parser.add_argument("--epoch", type=int, default=5, help="number of epoches of training")
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--valid_size', type=int, default=1000, help='input valid size') # 769

    alpha_list = [0.25, 0.5, 0.75]          # 3
    gamma_list = [0.5, 1, 2, 5]             # 4
    threshold_list = [0.15, 0.248, 0.4]     # 3

    parser.add_argument("--action", type=str, default='load', choices=['load', 'new', 'sample'], help="action to load or generate new features")
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

    # get dataset 
    trainset = Features(data_type='train', action=opt.action, feature_fname='FeatureOrigin')
    valset = Features(data_type='val', action=opt.action, feature_fname='FeatureOrigin')
    print('rows in trainset:', len(trainset)) # Should print 1217428
    print('rows in valset:', len(valset)) # Should print 304358

    # Use the torch dataloader to iterate through the dataset
    trainset_loader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)
    valset_loader = DataLoader(valset, batch_size=opt.valid_size, shuffle=False)

    device = get_device()
    model = Net().to(device) # Remember to move the model to "device"
    print(model)

    if opt.train_type == 'tune':
        print('Start Tuning Process ...')
        for threshold in range(3):
            for gamma in range(4):
                for alpha in range(3):
                    opt = args_parse(a=alpha, g=gamma, t=threshold)
                    print('\n\n\nStart Tuning Focal_a{}_g{}_t{} :\n'.format(str(alpha), str(gamma), str(threshold)))
                    train_save(model, trainset_loader, valset_loader, opt, epoch=opt.epoch, save_interval=5000, log_interval=200, device=device)
    elif opt.train_type == 'train':
        print('Start Training ...\n')
        train_save(model, trainset_loader, valset_loader, opt, epoch=opt.epoch, save_interval=5000, log_interval=200, device=device)
    elif opt.train_type == 'sample':
        # get some random training samples
        dataiter = iter(trainset_loader)
        features, labels = dataiter.next()
        print('Feature tensor in each batch:', features.shape, features.dtype)
        print('Label tensor in each batch:', labels.shape, labels.dtype)

