import numpy as np
from scipy.stats import binned_statistic
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import sys
from aemulusnu_massfunction.utils import *
box = sys.argv[1]
# box = 'Box_n50_0_1400'
# box = 'Box0_1400'
curr_run_fname = '/oak/stanford/orgs/kipac/aemulus/aemulus_nu/%s/'%(box)
rockstar_dir = curr_run_fname+'output/rockstar/'

f = open(rockstar_dir+'savelist.txt', 'r')
savelist = f.read().split()
f.close()

N_snapshots = len(savelist)

print(N_snapshots)

SNAPSHOT_IDX=0

import pickle

NvMs = {}
f = open('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_M200b', 'r')
TMP=0
skips = 0
for line in tqdm(f):
    print('\n\n')
    #extract the masses and position of halos for a given snapshot 
    snapshot_mass = line.strip().split()
    snapshot_mass = np.array(snapshot_mass, dtype=np.float64)


    #get the volume, redshift, and particle mass in the simulation
    vol = -1
    BOX_SIZE = -1
    a = -1
    Mpart = -1

    f_meta = open(rockstar_dir+'out_%d.list'%(SNAPSHOT_IDX), 'r')

    for meta_data in f_meta:
        if('#a' in meta_data):
            a = eval(meta_data.split()[2])
        if('Particle mass' in meta_data):
            Mpart = eval(meta_data.split()[2])
        if('Box size' in meta_data):
            vol = eval(meta_data.split()[2])**3
            BOX_SIZE = eval(meta_data.split()[2])
            break
    print('redshift', scaleToRedshift(a))
    SNAPSHOT_IDX+=1
    if(scaleToRedshift(a) >= 2.4): #skip the high redshift box
        print('skipping redshift', scaleToRedshift(a))
        skips+=1
        continue
#     gt200idxs[a] = gt200Np
    left_log10 = np.ceil(np.log10(200*Mpart) * 10) / 10 #e.g. 13.897234 -> 13.9
    print('leftlog10', left_log10)

    edges_log10 = np.arange(left_log10, 1+np.log10(np.max(snapshot_mass)), 0.1)
    edges = np.array([10**el10 for el10 in edges_log10])

    #get the number count of halos in the mass bins
    N, bin_edge, bin_idx = binned_statistic(snapshot_mass, np.ones_like(snapshot_mass),
                                            statistic='count', bins=edges)
    print('sumN', sum(N))
    if(sum(N) < 20):
        print('skipping redshift', scaleToRedshift(a))
        skips+=1
        continue

    print('N', N)
    print('edges', np.log10(edges))
    c_i = len(N)-1

    while(N[c_i] == 0):
        N = N[:c_i]
        bin_edge = bin_edge[:(c_i+1)]
        c_i -= 1

    #make large mass bin have at least 20 halos
    while(c_i >= 0 and N[c_i] < 20):
        N[c_i-1] += N[c_i]
        halos_here = np.where(bin_idx==c_i+1)
        bin_idx[halos_here] = c_i
        N = N[:c_i]
        bin_edge = np.delete(bin_edge,c_i)
        c_i -= 1
    if(c_i < 0):
        print('HUH?')
        assert(1==0)

    print('after adaptive', N)
    M_means = []
    correction = np.zeros_like(N)

    for j in range(len(N)):
        this_bin = np.where(bin_idx == j+1)
        M_means += [np.mean(snapshot_mass[this_bin])]
#         print(N[j], len(snapshot_mass[this_bin]))
        assert(len(snapshot_mass[this_bin]) == N[j])

    edge_pairs = [[bin_edge[j], bin_edge[j+1]] for j in range(len(bin_edge)-1)]
    assert(len(edge_pairs) == len(N))

    NvMs[a] = {'M':M_means, 
               'N':N, 
               'vol':vol, 
               'Mpart':Mpart, 
               'edge_pairs':edge_pairs,
               'bin_idx':bin_idx,
               'corrections':correction}
f.close()


NvM_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+curr_run_fname.split('/')[-2]+'_NvsM.pkl'
NvM_f = open(NvM_fname, 'wb')
pickle.dump(NvMs, NvM_f)
NvM_f.close()

jackknife = {}
f_pos = open('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_pos', 'r')
for _ in range(skips):
    f_pos.readline()
for a in tqdm(NvMs):
    snapshot_pos  = f_pos.readline().strip().split(',')
    snapshot_pos  = [np.array(pos.split(), dtype=np.float32) for pos in snapshot_pos if pos != '']
    snapshot_pos  = np.array(snapshot_pos)

#     gt200Np = gt200idxs[a]
    N = NvMs[a]['N']
    vol = NvMs[a]['vol']
    bin_idx = NvMs[a]['bin_idx']
    correction = NvMs[a]['corrections']
    assert(len(bin_idx) == len(snapshot_pos))

    #now lets get to spatial jackknife
    N_DIVS = 32 #each axis is diided into N_DIVS parts so in total the box
               #is divided into N_DIVS**3 boxes

    #compute the size of each smaller cube
    ϵ = vol*10**(-6)
    cube_vol = (vol+ε) / N_DIVS**3 #need ϵ to properly handle halos directly on boundary 
    cube_size = np.cbrt(cube_vol)
    rescale_factor = N_DIVS**3/(N_DIVS**3-1)

    #compute the indices of the smaller cube that each point belongs to
    cube_indices = (snapshot_pos / cube_size).astype(int)

    #cube_indices has assignment of halo to 3d position of a voxel
    #ravel_multi_index indexes the voxels in 3D with a single integer
    cube_assignment = np.ravel_multi_index(cube_indices.T, (N_DIVS, N_DIVS, N_DIVS), order='F')

    jackknife_data = []

    for i in range(N_DIVS**3):
        current_cube = np.where(cube_assignment == i)
        curr_N = np.zeros_like(N)
        for halo in bin_idx[current_cube]:
            #bin_idx=1 corresponds to first bin 
            #bin_idx-1 = idx of bin
            if(halo==0):
                continue
            curr_N[halo-1] += 1
        #get the histogram if we left out this sub-cube
        jackknife_data += [[a-b for (a,b) in zip(N, curr_N)]]

    jackknife_mean = np.mean(jackknife_data, axis=0)

    jackknife_data = np.array(jackknife_data) - jackknife_mean

    #arxiv: 0408569 eqn 6
    jackknife_covariance = [[rescale_factor * np.sum(jackknife_data.T[i] * jackknife_data.T[j], axis=0) for j in range(len(N))] for i in range(len(N))]
    jackknife_covariance = np.array(jackknife_covariance)

    jackknife[a] = [jackknife_data, jackknife_covariance]
f_pos.close()

jackknife_covs_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+curr_run_fname.split('/')[-2]+'_jackknife_covs.pkl'
jackknife_covs_f = open(jackknife_covs_fname, 'wb')
pickle.dump(jackknife, jackknife_covs_f)
jackknife_covs_f.close()
