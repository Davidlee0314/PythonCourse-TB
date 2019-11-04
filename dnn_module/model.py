import torch
import torch.nn as nn

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
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 16),
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

if __name__ == "__main__":
    model = Net()

    # x = torch.zeros(64, 384)
    # out = model(x)
    # print(out.shape)
    label = torch.tensor([[3],[0],[0],[8]])
    onehot = torch.zeros(4, 10).scatter_(1, label, 1)
    print(label.shape)
    print(onehot.shape)
    print(onehot)
