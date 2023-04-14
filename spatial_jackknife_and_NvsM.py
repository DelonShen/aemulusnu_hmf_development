import numpy as np
from scipy.stats import binned_statistic
from tqdm import tqdm, trange
import seaborn
import matplotlib.pyplot as plt
import os
import sys


box = sys.argv[1]
curr_run_fname = '/oak/stanford/orgs/kipac/aemulus/aemulus_nu/%s/'%(box)
rockstar_dir = curr_run_fname+'output/rockstar/'

f = open(rockstar_dir+'savelist.txt', 'r')
savelist = f.read().split()
f.close()

N_snapshots = len(savelist)

i=0

import pickle

NvMs = {}
f = open('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_M200b', 'r')

TMP=0
for line in tqdm(f):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))

    #extract the masses and position of halos for a given snapshot 
    snapshot_mass = line.strip().split()
    snapshot_mass = np.array(snapshot_mass, dtype=np.float64)  
    
    

    f = open(rockstar_dir+'out_%d.list'%(i), 'r')
    
    #get the volume, redshift, and particle mass in the simulation
    vol = -1
    BOX_SIZE = -1
    a = -1
    Mpart = -1
    for line in f:
        if('#a' in line):
            a = eval(line.split()[2])
        if('Particle mass' in line):
            Mpart = eval(line.split()[2])
        if('Box size' in line):
            vol = eval(line.split()[2])**3
            BOX_SIZE = eval(line.split()[2])
            break
            
    
    
    nBins = 16
    
    #we'll only consider halos with more than 100 particles
    edges = np.logspace(np.log10(100*Mpart), np.log10(np.max(snapshot_mass)), nBins, 10.)
    color = plt.colormaps["rainbow"]((i+1)/N_snapshots)[:-1]
    
    #get the number count of halos in the mass bins
    N, bin_edge, bin_idx = binned_statistic(snapshot_mass, np.ones_like(snapshot_mass), 
                                            statistic='count', bins=edges)
    bin_cnters = np.array([np.sqrt(bin_edge[i]*bin_edge[i+1]) for i in range(len(bin_edge)-1)])
    edge_pairs = np.array([[edges[i], edges[i+1]] for i in range(len(edges)-1)])
    
    #we'll look at only the bins with more than 10 halos in them
    Ngt10 = np.where(N>=10)
    bin_cnters = bin_cnters[Ngt10]
    edge_pairs = edge_pairs[Ngt10]
    N = N[Ngt10]
    
    #redefine the edges that we'll jackknife on 
    edges = [edge[0] for edge in edge_pairs]
    edges += [edge_pairs[-1][1]]    
    
    ax.scatter(bin_cnters, N, s=50, marker='x', c=color)
    ax.bar(x=edges[:-1], height=N, width=np.diff(edges), align='edge', fill=False, ec=color, label=r'$a=%.2f$'%(a))

    ax.set_title(curr_run_fname.split('/')[-2])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Mass $[h^{-1}M_\odot]$')
    ax.set_ylabel(r'$N$')
    ax.legend(frameon=False)

    plt.savefig('figures/'+curr_run_fname.split('/')[-2]+'_NvsM_a%.1f.pdf'%(a), bbox_inches='tight')

    i+=1
    assert(len(bin_cnters) == len(edge_pairs))
    assert(len(edge_pairs) == len(N))
    NvMs[a] = {'M':bin_cnters, 'N':N, 'vol':vol, 'Mpart':Mpart, 'edge_pairs':edge_pairs, 'bin_idx':bin_idx, 'Ngt10':Ngt10}

    
f.close()

NvM_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+curr_run_fname.split('/')[-2]+'_NvsM.pkl'
NvM_f = open(NvM_fname, 'wb')
pickle.dump(NvMs, NvM_f)
NvM_f.close()

jackknife = {}

f_pos = open('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_pos', 'r')


for a in NvMs:
    snapshot_pos  = f_pos.readline().strip().split(',')
    snapshot_pos  = [np.array(pos.split(), dtype=np.float32) for pos in snapshot_pos if pos != '']
    snapshot_pos  = np.array(snapshot_pos)

    bin_cnters = NvMs[a]['M']
    N = NvMs[a]['N']
    vol = NvMs[a]['vol']
    Mpart = NvMs[a]['Mpart']
    edge_pairs = NvMs[a]['edge_pairs']
    bin_idx = NvMs[a]['bin_idx']
    Ngt10 = NvMs[a]['Ngt10']
    print(a, np.min(bin_idx), np.max(bin_idx))
    print(len(bin_idx), len(N))
    #redefine the edges that we'll jackknife on 
    edges = [edge[0] for edge in edge_pairs]
    edges += [edge_pairs[-1][1]]    

    #now lets get to spatial jackknife
    N_DIVS = 8 #each axis is diided into N_DIVS parts so in total the box
               #is divided into N_DIVS**3 boxes

    #compute the size of each smaller cube
    ϵ = vol*10**(-6)
    cube_vol = (vol+ε) / N_DIVS**3 #need ϵ to properly handle halos directly on boundary 
    cube_size = np.cbrt(cube_vol)

    #compute the indices of the smaller cube that each point belongs to
    cube_indices = (snapshot_pos / cube_size).astype(int)

    #cube_indices has assignment of halo to 3d position of a voxel
    #ravel_multi_index indexes the voxels in 3D with a single integer
    cube_assignment = np.ravel_multi_index(cube_indices.T, (N_DIVS, N_DIVS, N_DIVS), order='F')
    
    jackknife_data = []
    
    idx_to_bin = dict(zip(np.array(Ngt10)[0], range(len(N))))
    print(len(cube_assignment), len(bin_idx))
    for i in trange(N_DIVS**3):
        current_cube = np.where(cube_assignment == i)
        curr_N = np.zeros_like(N)
        for halo in bin_idx[current_cube]:
            #bin_idx=1 corresponds to first bin (e.g. 0 in idx_to_bin)
            #bin_idx-1 = idx of bin
            if(halo-1 not in idx_to_bin.keys()): 
                continue
            curr_N[idx_to_bin[halo-1]] += 1
        #get the number count of halos in the mass bins
        jackknife_data += [[a-b for (a,b) in zip(N, curr_N)]]
    jackknife_data = np.array(jackknife_data).T
    jackknife_covariance = np.cov(jackknife_data)
    print(len(N), jackknife_data.shape, jackknife_covariance.shape)
    jackknife[a] = [jackknife_data, jackknife_covariance]
    
f_pos.close()

jackknife_covs_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+curr_run_fname.split('/')[-2]+'_jackknife_covs.pkl'
jackknife_covs_f = open(jackknife_covs_fname, 'wb')
pickle.dump(jackknife, jackknife_covs_f)
jackknife_covs_f.close()