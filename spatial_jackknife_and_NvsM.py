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
    NvMs[a] = {'M':bin_cnters, 'N':N, 'vol':vol, 'Mpart':Mpart, 'edge_pairs':edge_pairs, 'bin_idx':bin_idx}

    
f.close()

NvM_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+curr_run_fname.split('/')[-2]+'_NvsM.pkl'
NvM_f = open(NvM_fname, 'wb')
pickle.dump(NvMs, NvM_f)
NvM_f.close()

jackknife_NEW = {}

tot_bin_idx = []
tot_N = []
offsets = {}
for a in tqdm(NvMs):
    offsets[a] = len(tot_N)
    tot_N += [n for n in NvMs[a]['N']]
    tot_bin_idx += [bi+offsets[a]-1 for bi in NvMs[a]['bin_idx'] if bi != 0] #if bi=0 then mass below min mass threshold
tot_bin_idx = np.array(tot_bin_idx)
tot_N = np.array(tot_N)
print(tot_bin_idx.shape, tot_N.shape)
    
bin_counts = []


N_subsamples = int(2**19)

#compute the indices of the smaller cube that each point belongs to
shuffled = np.copy(tot_bin_idx)
np.random.shuffle(shuffled)

sample_size = len(shuffled) // N_subsamples  # Number of points in each subsample

for i in trange(N_subsamples):
    curr_N = np.zeros_like(tot_N)
    start_idx = i * sample_size
    end_idx = start_idx + sample_size
    if i == N_subsamples - 1:
        end_idx = len(shuffled)  # For the last subsample, adjust end index to include remaining points
    for halo in shuffled[start_idx:end_idx]:
        curr_N[halo] += 1
    #get the number count of halos in the mass bins when leaving out this subsample
    bin_counts += [tot_N-curr_N]

# Calculate the mean mass histogram over all random partitions
mean_histogram = np.mean(bin_counts, axis=0)

# Calculate the deviations from the mean for each mass bin for each random partition
deviations = bin_counts - mean_histogram
print(np.shape(deviations))
# Calculate the covariance matrix using the deviations from the mean for all random partitions
covariance = np.cov(deviations.T)

correction_factor =  N_subsamples/(N_subsamples - 1) 
covariance *= correction_factor

jackknife_covs_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+curr_run_fname.split('/')[-2]+'_jackknife_covs.pkl'
jackknife_covs_f = open(jackknife_covs_fname, 'wb')
pickle.dump(covariance, jackknife_covs_f)
jackknife_covs_f.close()