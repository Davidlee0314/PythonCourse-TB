import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(384, 128, kernel_size=5),
        #     nn.MaxPool1d(2),
        #     nn.ReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(128, 64, kernel_size=5),
        #     nn.Dropout2d(0.5),
        #     nn.MaxPool1d(2),
        #     nn.ReLU()
        # )
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
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fc5 = nn.Linear(16, 2)

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

        # soft max   >>  CE loss 自動轉 target 成為 one hot ，因此不需要 softmax 

        return x

if __name__ == "__main__":
    model = Net()

    x = torch.Tensor(4, 534).uniform_(0, 1)
    # x = torch.zeros(64, 534)
    output = model(x)
    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    print(output.shape)
    print(output)
    print(pred.shape)
    print(pred)

    softmax = nn.Softmax()
    sm = softmax(output).cpu() > 0.5
    print(sm.shape)
    print(sm)

    # pred = pred.float()
    # combine = torch.cat([output, pred], dim=1)

    output = output.detach().numpy()
    sm = sm.detach().numpy()
    pred = pred.detach().numpy()
    combine = np.concatenate((output, pred), axis=1)

    print(sm[:, 1])
    sm = sm[:, 1]
    

    # np.savetxt('output.csv', output, delimiter=',', fmt='%0.4f', header='txkey,output')
    np.savetxt('softmax.csv', sm, delimiter=',', fmt='%0.4f', header='txkey,softmax')
    # np.savetxt('pred.csv', pred, delimiter=',', fmt='%0.4f', header='txkey,pred')
    # np.savetxt('pred.csv', pred, delimiter=',', fmt='%d', header='txkey,pred')
    # np.savetxt('combine.csv', combine, delimiter=',', fmt='%d', header='txkey,combine')

    # label = torch.tensor([[3],[0],[0],[8]])
    # onehot = torch.zeros(4, 10).scatter_(1, label, 1)
    # print(label.shape)
    # print(onehot.shape)
    # print(onehot)
