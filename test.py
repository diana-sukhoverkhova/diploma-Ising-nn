import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.datasets as datasets
import os
import sys


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data(Jd, l, num_conf, T, num_temps, batch_size, shuffle_opt, opt='train'):
    datasets = []
    spins_loaded = np.load(f'data_spins/{Jd}_{opt}/spins_{l}_{T[0]}_{T[-1]}.npz')
    #spins_loaded = np.load(f'data_spins/{Jd}_{opt}/spins_{l}_{T[0]}_{T[-1]}_upd.npz')
    
    answ_loaded = np.load(f'data_spins/{Jd}_{opt}/answ_{l}_{T[0]}_{T[-1]}.npz')
    #answ_loaded = np.load(f'data_spins/{Jd}_{opt}/answ_{l}_{T[0]}_{T[-1]}_{opt_data}.npz')
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


def testing(model, test_dataloader, criterion, batch_size):
    outp = []
    errors = []
    accuracy = 0.0
    labels_all = []
    act = nn.Sigmoid()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.to(device)
            outputs = model(inputs)
            #outputs = act(outputs)
            outputs = outputs.squeeze(1)
            outp.append(act(outputs).numpy())
            loss = criterion(outputs, labels)
            errors.append(loss.item())
            
            accuracy += sum(abs(labels - act(outputs))).float().mean()
            
            labels_all.append(labels.numpy())
    
    accuracy = (1 - accuracy / len(test_dataloader)) * 100
    # print(np.array(labels_all)[:, :, 0], np.array(outp)[:, :, 0])
    mse = mean_squared_error(np.array(labels_all)[:,:,0], np.array(outp)[:,:,0], squared=False)
    logloss = log_loss(np.array(labels_all)[:,:,0], np.array(outp)[:,:,0])
    
    #print("Accuracy = {}".format(accuracy))
    #print("MSE = {}".format(mse))
    #print("LogLoss = {}".format(logloss))
    return outp, errors, labels_all, accuracy.item(), mse, logloss

def get_errs_outs(Jd, l, num_temps, epoch):
    
    # roots = [0.3018336135076354, 0.7883671029813581, 0.9723445185469877, 1.2390777517571931, 1.6410179299284857, 1.9728374883141215, 2.2691853142129728, 2.8007227202811613]
    # ms = [0.0004, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5]
    roots = [3.64095690650721,
        3.5184492410503965,
        3.393522159945615,
        3.2659503538230608,
        3.1354715107049764,
        3.0017774197047054,
        2.8645022087095424,
        2.7232065461969945,
        2.577356059664392,
        2.426291319035178,
        2.2691853142129728, 
        2.104982167992544, 
        1.932307699120554,
        1.749339162933206, 
        1.5536238493280832, 
        1.34187327905057, 
        1.109960313758399, 
        0.8541630993606272, 
        0.5762735442012712, 
        0.2885386111960936, 
        0.03198372863548067]
    jds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]

    get_crit_T = dict(zip(jds, roots))

    T_c = get_crit_T[Jd]
    T = np.round(np.linspace(T_c - 0.3, T_c + 0.3, num_temps), 4)
    
    num_conf_ts = 512

    criterion = nn.BCEWithLogitsLoss()     
    
    print(f'Start testing for L = {l}, Jd = {Jd}', flush=True)
    model = Net(l)
    
    T_c_ = 2.2691853142129728
    T_ = np.round(np.linspace(T_c_ - 0.3, T_c_ + 0.3, num_temps), 4)
    
    PATH = f'models/{l}_{0.0}_{T_[0]}_{T_[-1]}_{num_temps}_{epoch}_epochs.pt'

    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()
    test_dataloader = load_data(Jd, l, num_conf_ts, T, num_temps, batch_size=1, shuffle_opt=False, opt='test')
    outp, errors, labels, accuracy, mse, logloss = testing(model, test_dataloader, criterion, batch_size=1)
    # outp, errors, labels = testing(model, test_dataloader, criterion, batch_size=1)
    return errors, outp, labels, accuracy, mse, logloss


# roots = [0.3018336135076354, 0.7883671029813581, 0.9723445185469877, 1.2390777517571931, 1.6410179299284857, 1.9728374883141215, 2.2691853142129728, 2.8007227202811613]
# ms = [0.0004, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5]

roots = [3.64095690650721,
        3.5184492410503965,
        3.393522159945615,
        3.2659503538230608,
        3.1354715107049764,
        3.0017774197047054,
        2.8645022087095424,
        2.7232065461969945,
        2.577356059664392,
        2.426291319035178,
        2.2691853142129728, 
        2.104982167992544, 
        1.932307699120554,
        1.749339162933206, 
        1.5536238493280832, 
        1.34187327905057, 
        1.109960313758399, 
        0.8541630993606272, 
        0.5762735442012712, 
        0.2885386111960936, 
        0.03198372863548067]
jds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
get_crit_T = dict(zip(jds, roots))

num_epochs = int(sys.argv[1])
L = [20, 30, 60, 80]

metrics = {}
for l in L:
    metrics[l] = []

Jds = [1.0]
num_temps = int(sys.argv[2])
opt_data = sys.argv[3]
for Jd in Jds: 
    for l in L:
        errs_outs = get_errs_outs(Jd, l, num_temps, num_epochs)
        metrics[l].append([Jd, errs_outs[3], errs_outs[4], errs_outs[5]])
        
        np.save(f'data_errors/{Jd}_{l}_{num_temps}_{num_epochs}_epochs.npy', errs_outs[0])
        np.save(f'data_outputs/{Jd}_{l}_{num_temps}_{num_epochs}_epochs.npy', errs_outs[1])
        
        # np.save(f'data_errors/rectangular/{Jd}_{l}_{num_temps}_{num_epochs}_epochs.npy', errs_outs[0])
        # np.save(f'data_outputs/rectangular/{Jd}_{l}_{num_temps}_{num_epochs}_epochs.npy', errs_outs[1])
        
#path = f'metrics/rectangular/{num_epochs}_epoch_{opt_data}.npy'

path = f'metrics/tr_geq_zero/{num_epochs}_epoch.npy'
np.save(path, metrics)
