import numpy as np
import json
import os
import sys
from data_simulate import simulate

def smlt(opt, num_conf, L, Jd, Jv, T_):
    print('Start of generation:', L, Jd, T_, flush=True)
    path = f'data_spins/{Jd}_{opt}/spins_{L}_{T_}.npy'
    if not os.path.isfile(path):        
        # new_path = f'data_spins/{Jd}_{opt}/spins_{L}_{T_}/'
        # if not os.path.isdir(new_path):
        #     os.mkdir(new_path)
        
        spins = np.zeros((num_conf, L, L))
        answ = np.zeros((num_conf, 2))

        # always: Jh == 1.0 
        # if square lattice (without diagonal interactions): Jd == 0.0
        # if triangular lattice: Jv == 1.0
        sml = simulate(L, T_, 1, Jd, Jv, int(2*L**2.15), num_conf, path)
        
        for i in range(num_conf):
            spins[i] = np.reshape(sml[i], (-1, L))
            answ[i, 0] = float(get_crit_T[Jd] < T_)
            answ[i, 1] = float(get_crit_T[Jd] >= T_)
        
        np.save(f'data_spins/{Jd}_{opt}/spins_{L}_{T_}.npy', spins)
        np.save(f'data_spins/{Jd}_{opt}/answ_{L}_{T_}.npy', answ)

    print('End of generation')
        

with open('get_crit_T.json') as f:
  get_crit_T = json.loads(f.read())
  get_crit_T = {float(k): v for k,v in get_crit_T.items()}

with open('params.json') as g:
  params = json.loads(g.read())

L = params['L']
Jd = params['Jd']
Jv = params['Jv']
T_c = get_crit_T[Jd]
num_temps =  params['num_temps']
T = np.round(np.linspace(T_c - 0.3, T_c + 0.3, num_temps), 4)
# T = np.round(np.linspace(T_c-10**-2.0, T_c+10**-2.0, num_temps), 5)
# T = np.round(np.linspace(0.03, 3.5, num_temps), 4)
idx = int(sys.argv[1])
T_ = T[idx]
num_conf = params['num_conf']
opt = params['opt']

smlt(opt, num_conf, L, Jd, M, T_)
