#!/usr/bin/env python
# coding: utf-8


import numpy as np

deltat = 0.005 # us - depends on the size of the detector, namely on distance between two optical planes (or ~ twice the distance to a reflector in case of a single plane) 
period = 1000000 # us 
numdet = 10000 # pdms 
rate   = 200./1000000 # dark counts/1s /pdm

ngen = int(rate*period*numdet)

rng = np.random.default_rng(12345)
ids = np.arange(0,ngen)
dees = rng.integers(0,numdet,ngen)
tees = rng.uniform(0.,period,ngen)
#moms = np.full(ngen,-1)
primaries = np.column_stack((ids,dees,tees))
pict = 0.5
acc = 0.02
nptog = 3.5
pect = acc*nptog
#print(primaries)

def add_photons(phin):
    
    if (1):    # borel
        icts = rng.permutation(phin)[0:rng.poisson(pict*len(phin))]        
        ects = rng.permutation(phin)[0:rng.poisson(pect*len(phin))]
    else:      # geometric
        icts = rng.permutation(phin)[0:int(pict*len(phin))]        
        ects = rng.permutation(phin)[0:int(pect*len(phin))]

    ects[:,[2]] = np.full(len(ects),deltat).reshape(-1,1) + ects[:,[2]]
    ects[:,[1]] = rng.integers(0,numdet,len(ects)).reshape(-1,1)

    
    phout = np.vstack((ects,icts))
    if (len(phout)>0):
        phsec = add_photons(phout)
        phout = np.vstack((phout,phsec))
    
    return phout


print('apa=',len(add_photons(primaries))/len(primaries))


