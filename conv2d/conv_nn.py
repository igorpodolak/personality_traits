import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from datasets import WindowedEEGDataset
import os
from datetime import datetime
import torch.nn.functional as F
from tqdm import tqdm
import math
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./logsdir')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
batch_size = 32
channels = 19
path = os.getcwd() + '\\data\\REST_standarized'


# Assuming that we are on a CUDA machine, this should print a CUDA device:

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d1 = nn.Conv1d(19, 19, 10)
        self.batch_norm0 = nn.BatchNorm1d(19)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 10), stride=(1, 2))
        self.batch_norm = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 15), stride=(1, 2))
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=(2, 20), stride=(1, 2))
        self.batch_norm3 = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(400, 250)
        self.fc2 = nn.Linear(250, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 30)
        self.fc5 = nn.Linear(30, 20)
        self.fc6 = nn.Linear(20, 5)

        self.dropout1 = nn.Dropout1d(0.25)
        self.dropout2d1 = nn.Dropout2d(0.25)
        self.dropout2d2 = nn.Dropout2d(0.25)
        self.dropout2d3 = nn.Dropout2d(0.25)

        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool1d = nn.MaxPool1d(6, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1d1(x))
        x = self.pool1d(x)
        x = self.dropout1(x)
        x = self.batch_norm0(x)
        x = torch.unsqueeze(x, 1)
        # x = torch.transpose(x, 0, 1)
        x = F.relu(self.conv1(x))
        x = self.batch_norm(self.dropout2d1(self.pool(x)))
        x = self.batch_norm2(self.dropout2d2(self.pool(F.relu(self.conv2(x)))))
        x = self.batch_norm3(self.dropout2d3(self.pool(F.relu(self.conv3(x)))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


if __name__ == "__main__":
    dataset = WindowedEEGDataset(os.path.join(path, 'windowed_index.json'), path)  # TODO maybe needs more transforms
    splitted = train_val_dataset(dataset)
    trainloader = DataLoader(splitted['train'], batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(splitted['val'], batch_size=batch_size, shuffle=False, num_workers=4)

    net = Net()
    net.eval()
    inputs, labels = next(iter(valloader))
    writer.add_graph(net, inputs)
    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()
    if torch.cuda.is_available():
        net.cuda()
    summary(net, input_size=inputs[0].size())
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(6):  # loop over the dataset multiple times
        writer.flush()
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch * len(trainloader) + i)
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print(f'Training set: epoka: {epoch + 1}, iteracja {i + 1:5d} loss: {running_loss / 200 :.3f}')
                with open("test.txt", "a") as file_object:
                    # Append to file at the a
                    file_object.write(
                        f'Training set: epoka: {epoch + 1}, iteracja {i + 1:5d} loss: {running_loss / 200 :.3f} \n')
                running_loss = 0.0

        valid_loss = 0.0
        net.eval()
        for i, data in enumerate(valloader, 0):

            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/val", loss, epoch * len(valloader) + i)
            # print statistics
            valid_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print(f'Validation Set: epoka: {epoch + 1}, iteracja: {i + 1:5d} loss: {valid_loss / 200 :.3f}')
                with open("test.txt", "a") as file_object:
                    # Append to file at the a
                    file_object.write(
                        f'Validation Set: epoka: {epoch + 1}, iteracja {i + 1:5d} loss: {valid_loss / 200 :.3f} \n')
                valid_loss = 0.0

        now = datetime.now()
        date_time = now.strftime("_%Y-%m-%d_%H-%M")
        PATH = f'./TrainedModels/conv_nn{date_time}.pth'
        torch.save(net.state_dict(), PATH)
    writer.close()
    print('Finished Training')
