import platform
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from datasets import WindowedEEGDataset, WindowedSequenceEEGDataset
from datetime import datetime
import torch.nn.functional as F
from torch.nn import init

# writer = SummaryWriter('./logsdir')
from utils.loggers import Logger

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(139462371)

# todo zmienne do klas i funkcji jako pola klas oraz parametry!!!!
if platform.node().startswith('LAPTOP-0TK'):
    path = Path().absolute() / 'data' / 'REST_standardized'
elif platform.node().startswith('Igors-MacBook-Pro') or platform.node().startswith('igor-podolak-6.laptop.matinf'):
    DATA_ROOT_DIR = Path('/Users/igor/data')
    # info nazwa kartoteki z plikami -- wartosc w parametrze wywolania --datadir
    #     datadir = f"{DATA_ROOT_DIR}/personality_traits/RESTS_gr87"
    # datadir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87'
    standardized_dir = DATA_ROOT_DIR / 'personality_traits' / 'RESTS_gr87_standardized'
    path = standardized_dir


# check if in debug mode
def is_debug_mode() -> bool:
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False

# Assuming that we are on a CUDA machine, this should print a CUDA device:

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {
        'train': Subset(dataset, train_idx),
        'val': Subset(dataset, val_idx)
    }
    return datasets


class NetSeq(nn.Module):
    def __init__(self, fc1_size=250, nonlinearity_type='relu', init_mode='kaiming', kaiming_mode='fan_in'):
        super().__init__()
        self.fc1_size = fc1_size
        self.conv1d1 = nn.Conv1d(in_channels=19, out_channels=19, kernel_size=10)
        self.batch_norm0 = nn.BatchNorm1d(19)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 10), stride=(1, 2))
        self.batch_norm = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 15), stride=(1, 2))
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=40, kernel_size=(2, 20), stride=(1, 2))
        self.batch_norm3 = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(400, self.fc1_size)
        # self.fc2 = nn.Linear(250, 120)
        # self.fc3 = nn.Linear(120, 84)
        # self.fc4 = nn.Linear(84, 30)
        # self.fc5 = nn.Linear(30, 20)
        # self.fc6 = nn.Linear(20, 5)
        self.fc6 = nn.Linear(fc1_size, 5)

        self.dropout1 = nn.Dropout1d(0.25)
        self.dropout2d1 = nn.Dropout2d(0.25)
        self.dropout2d2 = nn.Dropout2d(0.25)
        self.dropout2d3 = nn.Dropout2d(0.25)

        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool1d = nn.MaxPool1d(6, stride=2)

        self.nonlinearity_type = nonlinearity_type

        self.init_mode = init_mode
        self.kaiming_mode = kaiming_mode
        if self.init_mode == 'kaiming':
            init.kaiming_uniform_(self.conv1d1.weight, mode=self.kaiming_mode, nonlinearity=self.nonlinearity_type)
            init.kaiming_uniform_(self.conv1.weight, mode=self.kaiming_mode, nonlinearity=self.nonlinearity_type)
            init.kaiming_uniform_(self.conv2.weight, mode=self.kaiming_mode, nonlinearity=self.nonlinearity_type)
            init.kaiming_uniform_(self.conv3.weight, mode=self.kaiming_mode, nonlinearity=self.nonlinearity_type)
        elif self.init_mode == 'xavier':
            gain_val = nn.init.calculate_gain(nonlinearity=self.nonlinearity_type)
            init.xavier_uniform_(self.conv1d1.weight, gain=gain_val)
            init.xavier_uniform_(self.conv1.weight, gain=gain_val)
            init.xavier_uniform_(self.conv2.weight, gain=gain_val)
            init.xavier_uniform_(self.conv3.weight, gain=gain_val)

        init.constant_(self.conv3.bias, 0)
        init.constant_(self.conv2.bias, 0)
        init.constant_(self.conv1.bias, 0)
        init.constant_(self.conv1d1.bias, 0)

        init.constant_(self.batch_norm0.weight, 1)
        init.constant_(self.batch_norm0.bias, 0)
        init.constant_(self.batch_norm.weight, 1)
        init.constant_(self.batch_norm.bias, 0)
        init.constant_(self.batch_norm2.weight, 1)
        init.constant_(self.batch_norm3.weight, 1)
        init.constant_(self.batch_norm2.bias, 0)
        init.constant_(self.batch_norm3.bias, 0)


        init.kaiming_uniform_(self.fc1.weight, mode=self.kaiming_mode, nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.kaiming_uniform_(self.fc6.weight, mode=self.kaiming_mode, nonlinearity='linear')
        init.constant_(self.fc6.bias, 0)

    def forward(self, z):
        y = torch.zeros(z.size()[0], z.size()[1], self.fc6.out_features)
        for k in range(z.size()[1]):
            x = z[:, k, ...]
            x = F.relu(self.conv1d1(x))
            x = self.batch_norm0(self.dropout1(self.pool1d(x)))
            x = torch.unsqueeze(x, 1)
            # x = torch.transpose(x, 0, 1)
            x = F.relu(self.conv1(x))
            x = self.batch_norm(self.dropout2d1(self.pool(x)))
            x = F.relu(self.conv2(x))
            x = self.batch_norm2(self.dropout2d2(self.pool(x)))
            x = F.relu(self.conv3(x))
            x = self.batch_norm3(self.dropout2d3(self.pool(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = self.fc6(x)
            y[:, k, :] = x
        out = torch.mean(y, 1)
        return out


if __name__ == "__main__":
    batch_size = 8
    num_workers = 2
    if is_debug_mode():
        num_workers = 1
    channels = 19
    save_models = False
    print_every = 0
    append_to_running_loss_file = False
    dataset_type = 'windowed'
    dataset_type = 'seq_windowed'

    # init_mode = 'xavier'
    init_mode = 'kaiming'
    kaiming_mode = 'fan_in'
    # kaiming_mode = 'fan_out'
    fc1_size = 100
    weight_decay = 0.5
    epochs_to_run = 100
    scheduler_type = "cosineLR"
    # scheduler_type = None

    do_logging = False
    do_logging = True
    if is_debug_mode():
        do_logging = False
    logger = Logger(flags={'neptune': do_logging}, project='personality-traits')
    logger.add(
        ["simple_convd", f"node={platform.node()}",
         f"data={dataset_type}",
         # f"spike_len={str(opts.n_spikes)}",
         f"batch={str(batch_size)}",
         f"init={init_mode}",
         f"kaiming_mode={kaiming_mode}",
         f"fc6_size={fc1_size}",
         f"decay={weight_decay:.1f}",
         f"epochs={epochs_to_run}",
         f"scheduler={scheduler_type}",
         # f"encoder={opts.encoder}", f"regularize={str(opts.regularize)}", f"lr={str(opts.learning_rate)}",
         # f"clip={str(opts.clip_grad)}", f"device={str(opts.device)}", f"num_features={opts.num_features}",
         # f"encode={opts.encode}", f"encode_scale={opts.encode_scale}", f"encode_len={opts.encode_len}",
         # f"seperated_branches={opts.seperated_branches}",
         # f"zero_state_after_example={opts.zero_state_after_example}",
         # f"regularize_coeff={opts.spike_regularization_coeff}",
         # f"correlation={opts.use_correlation_information}", f"contrastive={opts.contrastive_loss}"
         ])

    if dataset_type == 'windowed':
        dataset = WindowedEEGDataset(path / f'{dataset_type}_index.json', path)  # TODO maybe needs more transforms
    elif dataset_type == 'seq_windowed':
        dataset = WindowedSequenceEEGDataset(path / f'{dataset_type}_index.json', path)  # TODO maybe needs more transforms

    splitted = train_val_dataset(dataset, val_split=0.2)
    trainloader = DataLoader(splitted['train'], batch_size=batch_size, shuffle=True,
                             drop_last=True, num_workers=num_workers)
    valloader = DataLoader(splitted['val'],
                           # batch_size=len(splitted['val']),
                           batch_size=batch_size,
                           shuffle=False, drop_last=False, num_workers=num_workers)

    net = NetSeq(fc1_size=fc1_size, init_mode=init_mode, kaiming_mode=kaiming_mode)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), weight_decay=weight_decay)
    if scheduler_type is not None and scheduler_type.lower() == "cosinelr":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs_to_run - 1, verbose=True)
    for epoch in range(epochs_to_run):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            # todo why labels have size (number of examples, 1, 5)??? Why the 1?
            #  No problem in cnn_nn_npt, but a problem here. Solve it generally instead of a squeeze() here
            labels = torch.squeeze(labels, 1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            logger.log('loss', loss.item())
            running_loss += labels.size()[0] * loss.item()

        logger.log("loss-epoch", running_loss / len(trainloader.dataset))
        if scheduler_type is not None:
            scheduler.step()

        valid_loss = 0.0
        net.eval()
        # todo push ALL valid examples in one go, not in mini-batches
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            labels = torch.squeeze(labels, 1)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            valid_loss += labels.size()[0] * loss
        logger.log('valid', valid_loss / len(valloader.dataset))
        now = datetime.now()
        if save_models:
            date_time = now.strftime("_%Y-%m-%d_%H-%M")
            save_path = Path('./TrainedModels') / f'conv_nn{date_time}.pth'
            torch.save(net.state_dict(), save_path)

    logger.stop()
    print('Finished Training')
