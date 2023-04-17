from utils import *
import numpy as np
from scipy.stats import binned_statistic
from tqdm import tqdm, trange
import seaborn
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
import pickle

cosmos_f = open('cosmo_params.pkl', 'rb')
cosmo_params = pickle.load(cosmos_f) #cosmo_params is a dict
cosmos_f.close()

box = sys.argv[1]
h = cosmo_params[box]['H0']/100

Pk_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_Pk.pkl'
Pk_f = open(Pk_fname, 'rb')
Pkz = pickle.load(Pk_f) #Pkz is a dictonary of functions
Pk_f.close()

NvM_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_NvsM.pkl'
NvM_f = open(NvM_fname, 'rb')
NvMs = pickle.load(NvM_f) #NvMs is a dictionary of dictionaries
NvM_f.close()

#deal with floating point errors
a_to_z = dict(zip(NvMs.keys(), Pkz.keys()))
z_to_a = dict(zip(Pkz.keys(), NvMs.keys()))

# LOOKING_AT = [1]
from utils import *

N_data = {}
M_data = {}
from scipy.interpolate import interp1d

dlnσinvdMs = {}

vol = -1
Mpart = -1

for z in tqdm(Pkz.keys()):
    a = z_to_a[z]
#     if(a not in LOOKING_AT):
#         continue
    Pk = Pkz[z]
    c_data = NvMs[a]
    
    Ms = c_data['M'] #units of h^-1 Msolar
    N = c_data['N']
    edge_pairs = c_data['edge_pairs']
    assert(len(Ms) == len(edge_pairs))
    assert(len(Ms) == len(N))
    

    if(vol==-1):
        vol = c_data['vol']
    assert(vol == c_data['vol'])

    if(Mpart==-1):
        Mpart = c_data['Mpart']
    assert(Mpart == c_data['Mpart'])

    N_data[a] = N
    M_data[a] = Ms
    
    M_numerics = np.logspace(np.log10(100*Mpart), 17, 50) #h^-1 Msolar
    
    
    R = [M_to_R(m, box, a) for m in M_numerics] #h^-1 Mpc
    
    
    sigma2s = [sigma2(Pk, r) for r in R]
    sigma = np.sqrt(sigma2s)
    lnsigmainv = -np.log(sigma)
    dlnsinvdM = np.gradient(lnsigmainv, M_numerics)

    dσ2dR = [dsigma2dR(Pk, r) for r in R]
    dRdMs = [dRdM(m_c, box, a) for m_c in M_numerics]
    dlnσinvdM_2 = -1/2 *np.array([a/b*c for (a,b,c) in zip(dσ2dR, sigma2s, dRdMs)])
    
    f_dlnsinvdM_log = interp1d(np.log10(M_numerics), dlnsinvdM,kind='cubic')
    f_dlnsinvdM = lambda x: f_dlnsinvdM_log(np.log10(x))

    dlnσinvdMs[a] = f_dlnsinvdM
    
    f_M = np.logspace(np.log10(np.min(Ms)), np.log10(np.max(Ms)-1),100)

for a in N_data:
    N_data[a] = np.array(N_data[a])
    M_data[a] = np.array(M_data[a])

from scipy.special import gamma
from scipy.optimize import curve_fit
from utils import *

def p(a, p0, p1):
    oup = (p0)+(a-0.5)*(p1)
    return oup

def B(a, M, σM, d, e, f, g):
    oup = e**(d)*g**(-d/2)*gamma(d/2)
    oup += g**(-f/2)*gamma(f/2)
    oup = 2/oup
    return oup
    
    
def f_G(a, M, σM, d, e, f, g):
    oup = B(a, M, σM, d, e, f, g)
    oup *= ((σM/e)**(-d)+σM**(-f))
    oup *= np.exp(-g/σM**2)
    return oup
# d0 = 2.4
# f1 = 0.12
def tinker(a, M, 
           d0, d1, 
           e0, e1, 
           f0, f1,
           g0, g1,
           log10=False):
    d = p(a, d0, d1,)
    e = p(a, e0, e1,)
    f = p(a, f0, f1,)
    g = p(a, g0, g1,)
    
    R = M_to_R(M, box, a) #Mpc/h
    σM = np.sqrt(sigma2(Pkz[a_to_z[a]], R)) 
    oup = f_G(a, M, σM, d, e, f, g)
    oup *= rhom_a(box, a)/M
    oup *= dlnσinvdMs[a](M)
    if(log10):
        oup *= M*np.log(10)
    return oup/h**4 #h^4 to fix units

from utils import *

a_list = list(NvMs.keys())

from scipy.stats import poisson
param_names = [ 'd0', 'd1',
               'e0', 'e1',
               'f0', 'f1',
               'g0', 'g1',]
FIXED_VALS = {
# 'd0':d0,
# 'f1':f1,
}

def log_prior(param_values):
    #uniform prior
    params = dict(zip(param_names, param_values))
    for param in FIXED_VALS:
        params[param] = FIXED_VALS[param]
    for a in a_list:
        curr_params = [p(a, params['%s0'%l], params['%s1'%l]) for l in ['d','e','f','g']]
        for curr_param in curr_params:
            if(curr_param< 0 or curr_param>5):
                return -np.inf
    return 0

M_numerics = np.logspace(np.log10(100*Mpart), 17, 50)

jackknife_covs_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_jackknife_covs.pkl'
jackknife_covs_f = open(jackknife_covs_fname, 'rb')
jackknife = pickle.load(jackknife_covs_f)
jackknife_covs_f.close()

def calculate_inner_product(X, K_X):
    """
    Calculates -1/2 X^T (K_X)^(-1) X.
    
    Args:
        X (np.ndarray): Input array of shape (n,).
        K_X (np.ndarray): Input array of shape (n, n).
        
    Returns:
        float: Resulting scalar value.
    """
    # Ensure X and K_X have compatible shapes
    assert X.shape[0] == K_X.shape[0], "Number of rows in X must be equal to the number of rows in K_X"
    assert K_X.shape[0] == K_X.shape[1], "K_X must be a square matrix"
    
    # Calculate (K_X)^(-1)
    K_X_inv = np.linalg.inv(K_X)
    
    # Calculate X^T (K_X)^(-1) X
    inner_product = -0.5 * np.dot(np.dot(X.T, K_X_inv), X)
    
    return inner_product

def log_prob(param_values):   
    """
    Calculates the probability of the given tinker parameters 
    
    Args:
        param_values (np.ndarray): Input array of shape (number of params).
        
    Returns:
        float: Resulting log probability
    """

    if(log_prior(param_values) == -np.inf):
        return -np.inf
    
    params = dict(zip(param_names, param_values))
    tinker_fs = {}
    
    for a in N_data:
        tinker_eval = [tinker(a, M_c,**params,)*vol for M_c in M_numerics]
        f_dndlogM = interp1d(M_numerics, tinker_eval, kind='linear', bounds_error=False, fill_value=0.)
        tinker_fs[a] = f_dndlogM
        
    model_vals = {}
    for a in N_data:
        if(a_to_z[a] >=2):
#             print(1)
            continue
        model_vals[a] = np.array([quad(tinker_fs[a], edge_pair[0], edge_pair[1], epsabs=1e-1)[0]
            for edge_pair in NvMs[a]['edge_pairs']
        ])
    
        
    log_probs = [calculate_inner_product(model_vals[a]-N_data[a], jackknife[a][1]) for a in model_vals]
    if not np.isfinite(np.sum(log_probs)): 
        return -np.inf
    return np.sum(log_probs)

def log_likelihood(param_values):
    lp = log_prior(param_values)
    if not np.isfinite(lp):
        return -1e22
    return lp + log_prob(param_values)

from utils import *


result_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_MLFit.pkl'
result_f = open(result_fname, 'rb')
result = pickle.load(result_f)
result_f.close()

MLE_params = dict(zip(param_names, result['x']))

from scipy.interpolate import interp1d
nwalkers = 64
ndim = len(param_names)

initialpos = np.array([result['x'] for _ in range(nwalkers)]) + 1e-4 * np.random.normal(size=(nwalkers, ndim))

from multiprocessing import Pool

sampler = emcee.EnsembleSampler(
    nwalkers = nwalkers,
    ndim = ndim,
    log_prob_fn = log_likelihood,
    pool=Pool()
)

sampler.run_mcmc(initialpos, 1000, progress=True);

import corner
samples = sampler.chain[:, 750:, :].reshape((-1, ndim))

chain = sampler.flatchain
np.savetxt( "/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/" + box + "_MCMC_chain", chain)
likes = sampler.flatlnprobability
np.savetxt("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/" + box + "_MCMC_likes", likes)


params_final = dict(zip(param_names,np.percentile(samples,  50,axis=0)))

from scipy.interpolate import interp1d
i=0
for a in reversed(N_data.keys()):
    z = a_to_z[a]
    
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(13,16))
    plt.subplots_adjust(wspace=0, hspace=0)
    Pk = Pkz[z]
    c_data = NvMs[a]
    
    Ms = c_data['M']
    N = c_data['N']
    edge_pairs = c_data['edge_pairs']
    
    edges = [edge[0] for edge in edge_pairs]
    edges += [edge_pairs[-1][1]]



    dM = np.array([edges[1]-edges[0] for edges in edge_pairs])
    dndM = (N/vol)/dM

    tinker_eval = [tinker(a, M_c,**params_final,) for M_c in Ms]
    tinker_eval_ML = [tinker(a, M_c,**MLE_params,) for M_c in Ms]

    axs[1].plot(Ms, dndM, 'x-', color='black', label='Data')
    axs[1].plot(Ms, tinker_eval, 'o-', color='blue', label='Tinker ML+MCMC Fit')
    axs[1].plot(Ms, tinker_eval_ML, '+-', color='red', label='Tinker ML Fit')



    tinker_eval = [tinker(a, M_c,**params_final,)*vol for M_c in M_numerics]
    f_dndM = interp1d(M_numerics, tinker_eval, kind='linear', bounds_error=False, fill_value=0.)
    tinker_eval = np.array([quad(f_dndM, edge[0],  edge[1])[0] for edge in edge_pairs])

    
    tinker_eval_ML = [tinker(a, M_c,**MLE_params,)*vol for M_c in M_numerics]
    f_dndM = interp1d(M_numerics, tinker_eval_ML, kind='linear', bounds_error=False, fill_value=0.)
    tinker_eval_ML = np.array([quad(f_dndM, edge[0],  edge[1])[0] for edge in edge_pairs])

    color = plt.colormaps["rainbow"]((i+1)/len(Pkz.keys()))[:-1]

    edge_centers = [np.sqrt(edge[0]*edge[1]) for edge in edge_pairs]
    
    axs[0].scatter(Ms, N, s=50, marker='x', c='black')
    axs[0].scatter(edge_centers, tinker_eval, s=50 , marker='o', c='blue')
    axs[0].scatter(edge_centers, tinker_eval_ML, s=50 , marker='+', c='red')

    axs[0].bar(x=edges[:-1], height=N, width=np.diff(edges), align='edge', fill=False, ec='black', label='Data')
    axs[0].bar(x=edges[:-1], height=tinker_eval, width=np.diff(edges), align='edge', fill=False, ec='blue', label='Tinker ML+MCMC Fit')
    axs[0].bar(x=edges[:-1], height=tinker_eval_ML, width=np.diff(edges), align='edge', fill=False, ec='red', label='Tinker ML Fit')

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend(frameon=False)
    axs[0].set_ylabel('N')

    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].legend(frameon=False)
    axs[1].set_ylabel('N')
    axs[1].set_xlabel(r'Mass $[h^{-1}M_\odot]$')
    axs[1].set_ylabel(r'$dn/dM\ [{\rm Mpc}^{-3}M_\odot^{-1}]$')
    axs[0].set_title('%s, a=%.2f, z=%.2f'%(box, a,z))
    i+=1
    plt.savefig('figures/%s_ML+MCMCFits_a%.2f.pdf'%(box, a), bbox_inches='tight')