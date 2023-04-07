import numpy as np
from scipy.stats import binned_statistic
from tqdm import tqdm, trange
import seaborn
import matplotlib.pyplot as plt
import os
import sys

curr_run_fname = '/oak/stanford/orgs/kipac/aemulus/aemulus_nu/%s/'%(sys.argv[1])
rockstar_dir = curr_run_fname+'output/rockstar/'

f = open(rockstar_dir+'savelist.txt', 'r')
savelist = f.read().split()
f.close()

N_snapshots = len(savelist)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
i=0

import pickle

dndMs = {}
f = open('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+curr_run_fname.split('/')[-2]+'_M200b', 'r')
for line in tqdm(f):
    snapshot_mass = line.strip().split()
    snapshot_mass = np.array(snapshot_mass, dtype=np.float64)    
    f = open(rockstar_dir+'out_%d.list'%(i), 'r')
    
    vol = -1
    a = -1
    
    for line in f:
        if('#a' in line):
            a = eval(line.split()[2])
        if('Box size' in line):
            vol = eval(line.split()[2])**3
            break
    
    nBins = 30
    edges = np.logspace(np.log10(.9e11), np.log10(2e17), nBins, 10.)
    color = plt.colormaps["rainbow"]((i+1)/N_snapshots)
    N, bin_edge, bin_idx = binned_statistic(snapshot_mass, np.ones_like(snapshot_mass), 
                                            statistic='count', bins=edges)
    bin_cnters = np.array([np.sqrt(bin_edge[i]*bin_edge[i+1]) for i in range(len(bin_edge)-1)])
    dM = np.array([(bin_edge[i+1]-bin_edge[i]) for i in range(len(bin_edge)-1)])
    
    not0 = np.where(N>0)
    
    dndM = (N[not0]/vol)/dM[not0]

    ax.plot(bin_cnters[not0], dndM, c=color)
    ax.scatter(bin_cnters[not0], dndM, c=color,
              label=r'$a=%.2f$'%(a))
    i+=1
    dndMs[a] = {'M':bin_cnters[not0], 'dndM':dndM}
f.close()
ax.set_title(curr_run_fname.split('/')[-2])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Mass $[h^{-1}M_\odot]$')
ax.set_ylabel(r'$dn/dM$')
ax.legend(frameon=False)

plt.savefig('figures/'+curr_run_fname.split('/')[-2]+'_dndM.pdf', bbox_inches='tight')

dndM_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+curr_run_fname.split('/')[-2]+'_dndM.pkl'
dndM_f = open(dndM_fname, 'wb')
pickle.dump(dndMs, dndM_f)
dndM_f.close()