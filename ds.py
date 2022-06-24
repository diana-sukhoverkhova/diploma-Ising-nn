import numpy as np
import time 
import os
import sys
from tqdm import tqdm
from data_simulate import simulate


L = int(sys.argv[1])
roots = [2.2691853142129728, 2.104982167992544, 1.932307699120554, 1.749339162933206, 1.5536238493280832, 1.34187327905057, 1.109960313758399, 0.8541630993606272, 0.5762735442012712, 0.2885386111960936, 0.03198372863548067]
jds = [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
get_crit_T = dict(zip(jds, roots))

Jd = float(sys.argv[2])
T_c = get_crit_T[Jd]
num_temps = int(sys.argv[4])
#T = np.linspace(T_c - 0.3, T_c + 0.3, num_temps)
#T = np.round(T, 4)
T = np.round(np.linspace(0.03, 3.5, num_temps), 4)
idx = int(sys.argv[3])
T = T[idx]

def smlt():
    print('start gen', L, Jd, T, flush=True)
    if Jd == 0.0:
        num_conf = 2048
        opt = 'train'
        path = f'data_spins/{Jd}_{opt}/spins_{L}_{T}.npy'
        if not os.path.isfile(path):
            start = time.time()
            spins = np.zeros((num_conf, L, L))
            answ = np.zeros((num_conf))
            sml = simulate(L, T, 1, Jd, int(2*L**2.15), num_conf)

            for i in range(num_conf):
                spins[i] = np.reshape(sml[i], (-1, L))
                answ[i] = float(get_crit_T[Jd] < T)

            np.save(f'data_spins/{Jd}_train/spins_{L}_{T}.npy', spins)
            np.save(f'data_spins/{Jd}_train/answ_{L}_{T}.npy', answ)
            end = time.time()
            print('Training data was generated in', end - start, flush=True)

    num_conf = 512

    opt = 'test'
    path1 = f'data_spins/{Jd}_{opt}/spins_{L}_{T}.npy'
    path2 = f'data_spins/{Jd}_{opt}/answ_{L}_{T}.npy'
    if not os.path.isfile(path1) or not os.path.isfile(path2):
        start = time.time()
        spins = np.zeros((num_conf, L, L))
        answ = np.zeros((num_conf))
        sml = simulate(L, T, 1, Jd, int(2*L**2.15),num_conf)

        for i in range(num_conf):
            spins[i] = np.reshape(sml[i], (-1, L))
            answ[i] = float(get_crit_T[Jd] < T)

        np.save(f'data_spins/{Jd}_test/spins_{L}_{T}.npy', spins)
        np.save(f'data_spins/{Jd}_test/answ_{L}_{T}.npy', answ)
        end = time.time()
        print('Training data was generated in', end - start, flush=True)

smlt()
