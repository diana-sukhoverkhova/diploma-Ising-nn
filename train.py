import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.datasets as datasets
import os
import sys


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())

def load_data(Jd, l, num_conf, T, num_temps, batch_size, shuffle_opt, opt='train'):
    datasets = []
    spins_loaded = np.load(f'data_spins/{Jd}_{opt}/spins_{l}_{T[0]}_{T[-1]}.npz')
    answ_loaded = np.load(f'data_spins/{Jd}_{opt}/answ_{l}_{T[0]}_{T[-1]}.npz')
    for j in range(num_temps):
        
        #path = f'data_spins/{Jd}_{opt}/spins_{l}_{T[j]}.npy'
        #with open(path, 'rb') as f:
        #    x = np.load(f)   
        x = spins_loaded[f'T_{j}']
        tensor_x = torch.Tensor(x).unsqueeze(1)

        #path = f'data_spins/{Jd}_{opt}/answ_{l}_{T[j]}.npy'
        #with open(path, 'rb') as f:
        #    y = np.load(f)
        y = answ_loaded[f'T_{j}']
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
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.act_hid(x)
        x = x.view(-1, 64*int(l/2-1)*int(l/2-1))
        x = self.fc1(x)
        x = self.act_hid(x)
        x = self.fc2(x)
        return x


def train(l, train_dataloader, num_epochs, start_epoch, criterion, batch_size):
    act = nn.Sigmoid()

    T_c = get_crit_T[1.0]
    #T = np.round(np.linspace(T_c-10**-2.0, T_c+10**-2.0, num_temps), 5)
    T = np.round(np.linspace(T_c - 0.3, T_c + 0.3, num_temps), 4)
    #T = np.round(np.linspace(0.03, 3.5, num_temps), 4)

    for epoch in range(start_epoch, num_epochs+1):  
        model = Net(l)
        PATH = f'models/{l}_{Jd}_{T[0]}_{T[-1]}_{num_temps}_{epoch-1}_epochs.pt'

        if os.path.isfile(PATH):
            model.load_state_dict(torch.load(PATH))
            model.train()
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        PATH = f'models/{l}_{Jd}_{T[0]}_{T[-1]}_{num_temps}_{epoch-1}_epochs_opt.pt'
        if os.path.isfile(PATH):
            optimizer.load_state_dict(torch.load(PATH))

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        print(f'Start training for L = {l}, epoch = {epoch}', flush=True)
        running_loss = 0.0
        accuracy = 0.0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in pbar:
            inputs, labels = data
            #inputs = inputs.to(device)
            #labels = labels.to(device)

            inputs = inputs.cuda()
            labels = labels.cuda()  
            
            #model.to(device)
            model = model.cuda()
            
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

    print('Training completed', flush=True)
    return model

# roots = [2.2691853142129728, 2.104982167992544, 1.932307699120554, 1.749339162933206, 1.5536238493280832, 1.34187327905057, 1.109960313758399, 0.8541630993606272, 0.5762735442012712, 0.2885386111960936, 0.03198372863548067]
# jds = [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
# get_crit_T = dict(zip(jds, roots))

roots = [0.3018336135076354, 0.7883671029813581, 0.9723445185469877, 1.2390777517571931, 1.6410179299284857, 1.9728374883141215, 2.2691853142129728, 2.8007227202811613]
ms = [0.0004, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5]
get_crit_T = dict(zip(ms, roots))

def train_and_save(Jd, l, num_temps):
    num_conf_tr = 2048
    num_conf_ts = 512
    start_epoch = int(sys.argv[4])
    num_epochs = int(sys.argv[1])

    ####### change on Jd if it is needed to train on unsymmetric configurations ########### 
    T_c = get_crit_T[1.0]
    #T = np.round(np.linspace(T_c-10**-2.0, T_c+10**-2.0, num_temps), 5)
    T = np.round(np.linspace(T_c - 0.3, T_c + 0.3, num_temps), 4)
    #T = np.round(np.linspace(0.03, 3.5, num_temps), 4)
    criterion = nn.BCEWithLogitsLoss()     

    train_dataloader = load_data(Jd, l, num_conf_tr, T, num_temps, batch_size=4, shuffle_opt=True, opt='train')
    print(f'Start training for L = {l}', flush=True)
    model = train(l, train_dataloader, num_epochs, start_epoch, criterion, batch_size=4)

L = [int(sys.argv[2])]
Jd = 0.0
num_temps = int(sys.argv[3])
for l in L:
    train_and_save(Jd, l, num_temps)
