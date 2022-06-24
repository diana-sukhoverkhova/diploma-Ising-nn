import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.datasets as datasets
import os
import sys


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


def load_data(Jd, l, num_conf, T, num_temps, batch_size, shuffle_opt, opt='train'):
    datasets = []
    for j in range(num_temps):
        path = f'data_spins/{Jd}_{opt}/spins_{l}_{T[j]}.npy'
        with open(path, 'rb') as f:
            x = np.load(f)   
        tensor_x = torch.Tensor(x).unsqueeze(1)
        path = f'data_spins/{Jd}_{opt}/answ_{l}_{T[j]}.npy'
        with open(path, 'rb') as f:
            y = np.load(f)
        tensor_y = torch.from_numpy(y).type(torch.float32)

        datasets.append(TensorDataset(tensor_x, tensor_y))


    dataset = torch.utils.data.ConcatDataset(datasets)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_opt)


class Net(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.act_hid = nn.ReLU()
        self.fc1 = nn.Linear(64*int(l/2-1)*int(l/2-1), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.act_hid(x)
        x = x.view(-1, 64*int(l/2-1)*int(l/2-1))
        x = self.fc1(x)
        x = self.act_hid(x)
        x = self.fc2(x)
        return x


def train(l, train_dataloader, num_epochs, criterion, batch_size):
    act = nn.Sigmoid()

    T_c = get_crit_T[Jd]
    T = np.round(np.linspace(T_c - 0.3, T_c + 0.3, num_temps), 4)
    #T = np.round(np.linspace(0.03, 3.5, num_temps), 4)

    for epoch in range(1, num_epochs+1):  
        model = Net(l)
        PATH = f'models/{l}_0.0_{T[0]}_{T[-1]}_{num_temps}_{epoch-1}_epochs.pt'

        if os.path.isfile(PATH):
            model.load_state_dict(torch.load(PATH))
            model.train()
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        PATH = f'models/{l}_0.0_{T[0]}_{T[-1]}_{num_temps}_{epoch-1}_epochs_opt.pt'
        if os.path.isfile(PATH):
            optimizer.load_state_dict(torch.load(PATH))
        
        print(f'Start training for L = {l}, epoch = {epoch}')
        running_loss = 0.0
        accuracy = 0.0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in pbar:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)  
            
            model.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            #outputs = act(outputs)

            outputs = outputs.squeeze(1) # к одной размерности с labels

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            accuracy += (batch_size - sum(abs(labels - act(outputs)))).float().mean()

            pbar.set_description(
                    f"Loss: {running_loss/((i+1)*batch_size)} "
                    f"Accuracy: {accuracy * 100  / ((i+1)*batch_size)}"
            )
        
        PATH = f'models/{l}_{Jd}_{T[0]}_{T[-1]}_{num_temps}_{epoch}_epochs.pt'
        torch.save(model.state_dict(), PATH)
        PATH = f'models/{l}_{Jd}_{T[0]}_{T[-1]}_{num_temps}_{epoch}_epochs_opt.pt'
        torch.save(optimizer.state_dict(), PATH)

    print('Training completed')
    return model

roots = [2.2691853142129728, 2.104982167992544, 1.932307699120554, 1.749339162933206, 1.5536238493280832, 1.34187327905057, 1.109960313758399, 0.8541630993606272, 0.5762735442012712, 0.2885386111960936, 0.03198372863548067]
jds = [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
get_crit_T = dict(zip(jds, roots))

def train_and_save(Jd, l, num_temps):
    num_conf_tr = 2048
    num_conf_ts = 512
    num_epochs = int(sys.argv[1])

    T_c = get_crit_T[Jd]
    T = np.round(np.linspace(T_c - 0.3, T_c + 0.3, num_temps), 4)
    #T = np.round(np.linspace(0.03, 3.5, num_temps), 4)
    criterion = nn.BCEWithLogitsLoss()     

    train_dataloader = load_data(Jd, l, num_conf_tr, T, num_temps, batch_size=4, shuffle_opt=True, opt='train')
    print(f'Start training for L = {l}')
    model = train(l, train_dataloader, num_epochs, criterion, batch_size=4)

    #PATH = f'models/{l}_{Jd}_{T[0]}_{T[-1]}_{num_temps}_{num_epochs}_epochs.pt'
    #PATH = f'models/{l}_{Jd}_{T[0]}_{T[-1]}_{num_temps}.pt'
    #torch.save(model.state_dict(), PATH)

L = [int(sys.argv[2])]
Jd = 0.0
num_temps = int(sys.argv[3])
for l in L:
    train_and_save(Jd, l, num_temps)
