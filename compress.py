import numpy as np
import time 
import os
import sys
import json


def compress(L, Jd, T, num_temps, num_conf, opt):
    # create .npz file for 1 temperature 
    #opt = 'test'
    #num_temps = 100

    # T_c = get_crit_T[Jd]
    # T = np.round(np.linspace(T_c - 0.3, T_c + 0.3, num_temps), 4)
        
    print(f'Start compressing for L = {L}, Jd = {Jd}')

    xs = []
    for j in range(num_temps):
        path = f'data_spins/{Jd}_{opt}/spins_{L}_{T[j]}.npy'
        with open(path, 'rb') as f:
            x_j = np.load(f)
            xs.append(x_j)
        os.remove(path)

    savez_dict = dict()
    for j, x_j in enumerate(xs):
        savez_dict[f'T_{j}'] = x_j

    np.savez_compressed(f'data_spins/{Jd}_{opt}/spins_{L}_{T[0]}_{T[-1]}', **savez_dict)

    ys = []
    for j in range(num_temps):
        path = f'data_spins/{Jd}_{opt}/answ_{L}_{T[j]}.npy'
        with open(path, 'rb') as f:
            y_j = np.load(f)
            ys.append(y_j)
        os.remove(path)

    savez_dict = dict()
    for j, y_j in enumerate(ys):
        savez_dict[f'T_{j}'] = y_j

    np.savez_compressed(f'data_spins/{Jd}_{opt}/answ_{L}_{T[0]}_{T[-1]}', **savez_dict)
    print('Compression is done')


with open('get_crit_T.json') as f:
  get_crit_T = json.loads(f.read())
  get_crit_T = {float(k): v for k,v in get_crit_T.items()}

with open('params.json') as g:
  params = json.loads(g.read())

L = params['L']
Jd = params['Jd']
T_c = get_crit_T[Jd]
num_temps =  params['num_temps']
T = np.round(np.linspace(T_c - 0.3, T_c + 0.3, num_temps), 4)
# T = np.round(np.linspace(T_c-10**-2.0, T_c+10**-2.0, num_temps), 5)
# T = np.round(np.linspace(0.03, 3.5, num_temps), 4)
num_conf = params['num_conf']
opt = params['opt']
compress(L, Jd, T, num_temps, num_conf, opt)