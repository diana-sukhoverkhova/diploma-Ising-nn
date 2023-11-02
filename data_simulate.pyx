cimport cython
import numpy as np
cimport numpy as cnp

from libc.math cimport exp, tanh
from mc_lib.rndm cimport RndmWrapper
from mc_lib.lattices import tabulate_neighbors
from mc_lib.observable cimport RealObservable

import time
import os


cdef RndmWrapper rndm = RndmWrapper((1234, 0)) # global variable
    
cdef void init_spins(cnp.int32_t[::1] spins): 
    
    for j in range(spins.shape[0]):
        spins[j] = 1 if rndm.uniform() > 0.5 else -1
        
        
        
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double energy(cnp.int32_t[::1] spins, 
                   cnp.int32_t[:, ::1] neighbors,
                  const double[:,::1] Js):

    cdef:
        double ene = 0.0
        Py_ssize_t site, site1, num_neighb

    for site in range(spins.shape[0]):
        num_neighb = neighbors[site, 0]
        for j in range(1, num_neighb+1):
            site1 = neighbors[site, j]
            ene += -1 * Js[site, site1] * spins[site] * spins[site1] 
    
    return ene / 2.0

'''
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void tabulate_ratios(double[::1] ratios1,
                      double[::1] ratios2,
                      double beta,
                      int nDim1,
                      int nDim2, 
                      double Jd):
    cdef:
        int summ
    for summ in range(-nDim1, nDim1+1, 2):
        ratios1[summ + nDim1] = exp(-2.0*beta * summ)
    for summ in range(-nDim2, nDim2+1, 2):
        ratios2[summ + nDim2] = exp(-2.0*beta * summ * Jd)
'''

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void flip_spin(cnp.int32_t[::1] spins, 
                    cnp.int32_t[:, ::1] neighbors,
                    double beta,
                    const double[:,::1] Js): 
    cdef:
        Py_ssize_t site = int(spins.shape[0] * rndm.uniform())
        Py_ssize_t site1

    cdef int num_neighb = neighbors[site, 0]
    cdef double summ = 0.
    for j in range(1, num_neighb + 1):
        site1 = neighbors[site, j]
        summ += spins[site1] * spins[site] * Js[site,site1]
        
    cdef double ratio = exp(-2.0 * beta * summ )
    
    if ratio < 1:
        if rndm.uniform() > ratio:
            return

    spins[site] *= -1
    
    
cdef void get_J( double[:,::1] Js, double Jh, double Jd, double Jv, int L1, int L2 , int L3 = 1):
  
    if L3 == 1:
        for i in range(L1*L2):
            Js[i, ((i // L2 + 1) % L1 * L2 )  + (i + 1) % L2 ] = Jd
            Js[i, ((i // L2  - 1) % L1 * L2 )  + (i - 1) % L2 ] = Jd
            Js[i, (i // L2) * L2 + (i + 1) % L2] = Jv
            Js[i, (i + L2) % (L1*L2)] = Jh
            Js[i, (i // L2) * L2 + (i - 1) % L2] = Jv
            Js[i, (i - L2) % (L1*L2)] = Jh
        return
    
    else:
        return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
cdef _simulate(Py_ssize_t L,
             double T, double Jh, double Jd, double Jv,
             Py_ssize_t num_sweeps,
             Py_ssize_t num_conf, 
             str path):

    cdef: 
        double start = time.time()
        double end, tmp

    cdef:
        cnp.int32_t[:, ::1] neighbors = np.asarray(tabulate_neighbors(L, kind='triang'), np.int32)
        #cnp.int32_t[:, ::1] neighbors = np.asarray(tabulate_neighbors((L, L, 1), kind='sc'), np.int32)
        double beta = 1./T
 
    cdef:
        int num_therm = int(20*L**2.15)
        int steps_per_sweep = int(L * L)
        int sweep = 0
        int i
        # double av_en = 0., Z = 0., magn = 0., av_magn=0.

    cdef cnp.int32_t[::1] spins =  np.empty( L*L, dtype=np.int32) 
    init_spins(spins)
    
    cdef double[:,::1] Js = np.zeros((L*L, L*L)) 
    get_J(Js, Jh, Jd, Jv, L, L)

    cdef cnp.int32_t[:,::1] res = np.empty((num_conf, L*L), dtype=np.int32)

    for sweep in range(num_therm):
        for i in range(steps_per_sweep):
            flip_spin(spins, neighbors, beta, Js)
    end = time.time()
    print('Thermalization time : ', end - start)
    for conf in range(num_conf):
        tmp = time.time()
        for sweep in range(num_sweeps):
            for i in range(steps_per_sweep):
                flip_spin(spins, neighbors, beta, Js)
        end = time.time()
        print('One configuration time:', end - tmp, 'conf = ', conf)
        res[conf, :] = spins
        #new_path = path + str(conf) + '.npy'
        #if not os.path.isfile(new_path):
        #    np.save(new_path, spins)
    
    end = time.time()
    print('Total time: ', end - start)
        
    return (res)

def simulate(L, T, Jh, Jd, Jv, num_sweeps, num_conf, path):
    return _simulate(L, T, Jh, Jd, Jv, num_sweeps, num_conf, path)
