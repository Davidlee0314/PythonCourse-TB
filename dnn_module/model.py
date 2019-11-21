import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        # print('input x.shape :', x.shape)

        x = self.fc1(x)
        # print('fc1 x.shape :', x.shape)

        x = self.fc2(x)
        # print('fc2 x.shape :', x.shape)

        x = self.fc3(x)
        # print('fc3 x.shape :', x.shape)

        x = self.fc4(x)
        # print('fc4 x.shape :', x.shape)

        # soft max   >>  CE loss 自動轉 target 成為 one hot ，因此不需要 softmax 

        return x

class Net1D(nn.Module):
    def __init__(self):
        super(Net1D, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(534, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        # print('input x.shape :', x.shape)

        x = self.fc1(x)
        # print('fc1 x.shape :', x.shape)

        x = self.fc2(x)
        # print('fc2 x.shape :', x.shape)

        x = self.fc3(x)
        # print('fc3 x.shape :', x.shape)

        x = self.fc4(x)
        # print('fc4 x.shape :', x.shape)

        # soft max   >>  CE loss 自動轉 target 成為 one hot ，因此不需要 softmax 

        return x

class Net1D_2(nn.Module):
    def __init__(self):
        super(Net1D_2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(534, 256),
            
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 256),
            
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        self.fc4 = nn.Linear(256, 2)

    def forward(self, x):
        # print('input x.shape :', x.shape)

        x = self.fc1(x)
        # print('fc1 x.shape :', x.shape)

        x = self.fc2(x)
        # print('fc2 x.shape :', x.shape)

        x = self.fc3(x)
        # print('fc3 x.shape :', x.shape)

        x = self.fc4(x)
        # print('fc4 x.shape :', x.shape)

        # soft max   >>  CE loss 自動轉 target 成為 one hot ，因此不需要 softmax 

        return x


class Net1D_3(nn.Module):
    def __init__(self):
        super(Net1D_3, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(534, 256),
            
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 256),
            
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(128, 64),
            
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )
        self.fc6 = nn.Linear(64, 2)

    def forward(self, x):
        # print('input x.shape :', x.shape)

        x = self.fc1(x)
        # print('fc1 x.shape :', x.shape)

        x = self.fc2(x)
        # print('fc2 x.shape :', x.shape)

        x = self.fc3(x)
        # print('fc3 x.shape :', x.shape)

        x = self.fc4(x)
        # print('fc4 x.shape :', x.shape)

        x = self.fc5(x)
        # print('fc5 x.shape :', x.shape)

        x = self.fc6(x)
        # print('fc6 x.shape :', x.shape)
        # soft max   >>  CE loss 自動轉 target 成為 one hot ，因此不需要 softmax 

        return x


class Net2D(nn.Module):
    def __init__(self):
        super(Net2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=9, padding=1),
            nn.BatchNorm2d(4),
            # nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=9, padding=1),
            nn.BatchNorm2d(8),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=9, padding=1),
            nn.BatchNorm2d(16),
            # nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*9, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # print('input x.shape :', x.shape)

        x = self.conv1(x)
        # print('conv1 x.shape :', x.shape)

        x = self.conv2(x)
        # print('conv2 x.shape :', x.shape)
        
        x = self.conv3(x)
        # print('conv3 x.shape :', x.shape)
        
        x = x.view(-1, 16*9)
        # print('view x.shape :', x.shape)

        x = self.fc1(x)
        # print('fc1 x.shape :', x.shape)

        x = self.fc2(x)
        # print('fc2 x.shape :', x.shape)

        # soft max   >>  CE loss 自動轉 target 成為 one hot ，因此不需要 softmax 

        return x


class Net2D_2(nn.Module):
    def __init__(self):
        super(Net2D_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=21, padding=3),
            nn.BatchNorm2d(8),
            # nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=9, padding=3),
            nn.BatchNorm2d(16),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=7, padding=1),
            nn.BatchNorm2d(16),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # print('input x.shape :', x.shape)

        x = self.conv1(x)
        # print('conv1 x.shape :', x.shape)

        x = self.conv2(x)
        # print('conv2 x.shape :', x.shape)
        
        x = self.conv3(x)
        # print('conv3 x.shape :', x.shape)
        
        x = x.view(-1, 16*16)
        # print('view x.shape :', x.shape)

        x = self.fc1(x)
        # print('fc1 x.shape :', x.shape)

        x = self.fc2(x)
        # print('fc2 x.shape :', x.shape)

        # soft max   >>  CE loss 自動轉 target 成為 one hot ，因此不需要 softmax 

        return x


if __name__ == "__main__":
    # model = Net()
    model2D = Net2D_2()

    combine = torch.Tensor(4, 576).uniform_(0, 1)
    # x = torch.Tensor(534).uniform_(0, 1)
    # zero = torch.zeros(42)
    # combine = torch.cat([x , zero])
    
    print(combine.shape)
    combine = combine.unsqueeze(0)
    combine = combine.view(-1, 1, 24, 24)
    print(combine.shape)
    # print(combine)

    output = model2D(combine)
    print(output.shape)
    print(output)

    softmax = nn.Softmax()
    sm = softmax(output)
    print(sm.shape)
    print(sm)
    
    # output = model(x)
    # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    # softmax = nn.Softmax()
    # sm = softmax(output).cpu() > 0.5
    # print(sm.shape)
    # print(sm)

    # output = output.detach().numpy()
    # sm = sm.detach().numpy()
    # pred = pred.detach().numpy()
    # combine = np.concatenate((output, pred), axis=1)
    # print(sm[:, 1])
    # sm = sm[:, 1]
    
    # # np.savetxt('output.csv', output, delimiter=',', fmt='%0.4f', header='txkey,output')
    # np.savetxt('softmax.csv', sm, delimiter=',', fmt='%0.4f', header='txkey,softmax')
    # # np.savetxt('pred.csv', pred, delimiter=',', fmt='%0.4f', header='txkey,pred')
    # # np.savetxt('pred.csv', pred, delimiter=',', fmt='%d', header='txkey,pred')
    # # np.savetxt('combine.csv', combine, delimiter=',', fmt='%d', header='txkey,combine')

