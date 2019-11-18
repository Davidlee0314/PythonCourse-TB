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

class Net2D(nn.Module):
    def __init__(self):
        super(Net2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, padding=1),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        print('input x.shape :', x.shape)

        x = self.conv1(x)
        print('conv1 x.shape :', x.shape)

        x = self.conv2(x)
        print('conv2 x.shape :', x.shape)
        
        x = x.view(-1, 320)
        print('view x.shape :', x.shape)

        x = self.fc1(x)
        print('fc1 x.shape :', x.shape)

        x = self.fc2(x)
        print('fc2 x.shape :', x.shape)

        # soft max   >>  CE loss 自動轉 target 成為 one hot ，因此不需要 softmax 

        return x

if __name__ == "__main__":
    # model = Net()
    model2D = Net2D()

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

