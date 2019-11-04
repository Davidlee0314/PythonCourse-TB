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

def lgb_f1_score(y_hat, data, THRESHOLD=0.248):
    y_true = data.get_label()
    y_hat = np.where(y_hat >= THRESHOLD, 1, 0)
    return 'f1', f1_score(y_true, y_hat), True

def eval(model, testset_loader, threshold=0.15):
    print('Start Evaluating ...')
    criterion = nn.CrossEntropyLoss()
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
    f1 = f1_score(labels_onehot.cpu(), output_all.sigmoid().cpu() > threshold, average='macro')

    test_loss /= len(testset_loader.dataset)
    print('\nEval set: \n\tAverage loss: {:.4f} \n\tAccuracy: {:.0f}% ({}/{}) \n\tF1 Score: {}\n'.format(
        test_loss, 100. * correct / len(testset_loader.dataset), correct, len(testset_loader.dataset), f1))

def train_save(model, trainset_loader, testset_loader, opt, epoch=5, save_interval=500, log_interval=100, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

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
                save_checkpoint('./models/NetOrigin_backup.pth', model, optimizer)
            iteration += 1
        save_checkpoint('./models/NetOrigin_epoch_{}.pth'.format(epoch), model, optimizer)
        eval(model, testset_loader)
    
    # save the final model
    save_checkpoint('./models/NetOrigin_final.pth' % iteration, model, optimizer)

def get_device():
    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    return device

def args_parse():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--output_folder', default='DCGAN_output', help='folder to output images and model checkpoints')
    parser.add_argument("--epoch", type=int, default=5, help="number of epoches of training")
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--valid_size', type=int, default=1000, help='input valid size') # 769

    # parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    # parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    # parser.add_argument('--model_path', default='', help="path to model (to continue training)")
    # parser.add_argument('--load_features', action='store_true', help="used when training (saveing time to run resnet, will load pre-processed CNN_features of videos [frames, 3, 224, 224])")

    opt = parser.parse_args()
    print(opt)
    return opt

if __name__ == '__main__':
    # parse training args
    opt = args_parse()

    # get dataset 
    trainset = Features(data_type='train', action='new', feature_fname='FeatureOrigin')
    valset = Features(data_type='val', action='new', feature_fname='FeatureOrigin')
    print('rows in trainset:', len(trainset)) # Should print 1217428
    print('rows in valset:', len(valset)) # Should print 304358

    # Use the torch dataloader to iterate through the dataset
    trainset_loader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)
    valset_loader = DataLoader(valset, batch_size=opt.valid_size, shuffle=False)

    device = get_device()
    model = Net().to(device) # Remember to move the model to "device"
    print(model)

    print('Start Training ...\n')
    train_save(model, trainset_loader, valset_loader, opt, epoch=opt.epoch, save_interval=500, log_interval=100)




    # get some random training samples
    # dataiter = iter(trainset_loader)
    # features, labels = dataiter.next()
    # print('Feature tensor in each batch:', features.shape, features.dtype)
    # print('Label tensor in each batch:', labels.shape, labels.dtype)

