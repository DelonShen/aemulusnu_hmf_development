import sys


from aemulusnu_massfunction.utils import *
from aemulusnu_massfunction.massfunction import *


import numpy as np
from scipy import optimize as optimize


box = sys.argv[1]

print('##############################')
print()
print()


prev_box = sys.argv[2]



KX = np.diag([1e-2, 1e-2,
              1e-3, 1e-3,
              5e-4, 5e-4,
              1e-4, 1e-4])


param_names = ['d0','d1',
               'e0','e1',
               'f0','f1',
               'g0','g1']
ndim = len(param_names)


from aemulusnu_massfunction.utils import *
from aemulusnu_massfunction.massfunction import *

import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
import pickle

cosmos_f = open('../data/cosmo_params.pkl', 'rb')
cosmo_params = pickle.load(cosmos_f) #cosmo_params is a dict
cosmos_f.close()


import pyccl as ccl

cosmo = cosmo_params[box]


h = cosmo['H0']/100
Ωb =  cosmo['ombh2'] / h**2
Ωc =  cosmo['omch2'] / h**2

ccl_cosmo = ccl.Cosmology(Omega_c=Ωc,
                      Omega_b=Ωb,
                      h=h,
                      A_s=cosmo['10^9 As']*10**(-9),
                      n_s=cosmo['ns'],
                      w0=cosmo['w0'],
                      m_nu=[cosmo['nu_mass_ev']/3, cosmo['nu_mass_ev']/3, cosmo['nu_mass_ev']/3])


h = cosmo['H0']/100

NvM_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_NvsM.pkl'
NvM_f = open(NvM_fname, 'rb')
NvMs = pickle.load(NvM_f) #NvMs is a dictionary of dictionaries
NvM_f.close()



N_data = {}
M_data = {}
aux_data = {}


vol = -1 #Mpc^3/h^3
Mpart = -1

for a in tqdm(NvMs):
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

    N_data[a] = []
    M_data[a] = []
    aux_data[a] = []
    for N_curr, M_curr, edge_pair in zip(N, Ms, edge_pairs):
        N_data[a] += [N_curr]
        M_data[a] += [M_curr]
        aux_data[a] += [{'a':a, 'edge_pair':edge_pair}]


a_list = list(NvMs.keys())
print(a_list)
from scipy.stats import poisson


M_numerics = np.logspace(np.log10(100*Mpart), 16, 50)

jackknife_covs_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_jackknife_covs.pkl'
jackknife_covs_f = open(jackknife_covs_fname, 'rb')
jackknife = pickle.load(jackknife_covs_f)
jackknife_covs_f.close()

print(jackknife.keys())


jack_covs = {a:jackknife[a][1] for a in N_data}

# Inverse of the weighted covariance matrix
inv_weighted_cov = {a:np.linalg.inv(jack_covs[a]) for a in jack_covs}  

scale_cov = {a:np.log(np.linalg.det(jack_covs[a])) for a in jack_covs}


mass_function = MassFuncAemulusNu_fitting_all_snapshot()


def uniform_log_prior(param_values):
    #uniform priorb
    paired_params = list(zip(param_values, param_values[1:]))[::2]
    for i, (p0, p1) in enumerate(paired_params):
        for a in a_list:
            param = p(p0, p1, a)
            if(param < 0 or param > 5):
                return -np.inf

    #g = g0 + (a-0.5) g1
    # a smaller -> earlier time
    # want g to get bigger at earlier time
    # => g1 < 0
    # g bigger => exp suppression happens at smaller mass
    # so we should exclude g1 > 0
    if(param_values[-1] > 0 ): 
        return -np.inf
    return 0

def log_likelihood(param_values):
    """
    Calculates the likelihood of the given tinker parameters
    Args:
        param_values (np.ndarray): Input array of shape (number of params).
    Returns:
        float: Resulting log probability
    """

    if(uniform_log_prior(param_values) == -np.inf):
        return -np.inf

    params = dict(zip(param_names, param_values))

    tinker_fs = {}
    model_vals = {}

    mass_function.set_params(param_values)
    for a_fit in a_list:
        f_tinker_eval = lambda M:mass_function(ccl_cosmo, M/h, a_fit)*vol/(h**3 * M * np.log(10))
        tinker_fs[a_fit] = f_tinker_eval
        model_vals[a_fit] = np.array([quad(tinker_fs[a_fit], edge_pair[0], edge_pair[1], epsabs=0, epsrel=5e-3)[0]
            for edge_pair in NvMs[a_fit]['edge_pairs']
        ])

    residuals = {a: model_vals[a]-N_data[a] for a in model_vals}
    log_probs = [ -0.5 * (len(inv_weighted_cov[a])* np.log(2*np.pi) +
                          np.dot(np.dot(residuals[a].T, inv_weighted_cov[a]), residuals[a]) +
                          scale_cov[a])
                 for a in model_vals]
    if not np.isfinite(np.sum(log_probs)):
        return -np.inf
    return np.sum(log_probs)


def log_prob(param_values):
    lp = uniform_log_prior(param_values)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(param_values)



prev_params_final = None
with open("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/%s_params.pkl"%(prev_box), "rb") as f:
    prev_params_final = pickle.load(f)

prev_final_param_vals = np.array(list(prev_params_final.values()))
print('prev', prev_params_final)
KXinv = np.linalg.inv(KX)
detKX = np.linalg.det(KX)
logdetKX = np.log(detKX)




def log_prior(param_values):
    xmp = param_values - prev_final_param_vals
    arg = np.dot(np.dot(xmp.T, KXinv), xmp)
    return -1/2*(ndim * np.log(2*np.pi) + logdetKX + arg)


def log_prob_with_prior(param_values):
    lp = log_prior(param_values)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_prob(param_values)



guess = prev_final_param_vals


from scipy import optimize as optimize

print('Starting ML Fit')
#Start by sampling with a maximum likelihood approach
from scipy import optimize as optimize
neg_log_posterior = lambda *args: -log_prob_with_prior(*args)
result = optimize.minimize(neg_log_posterior, guess, 
                           method='Nelder-Mead',
                           options={'maxiter': len(guess)*10000})
result['param_names'] = param_names
print(box)
print(result)
print(result['x'])


MLE_params = dict(zip(param_names, result['x']))
print(MLE_params)


yerr_dict = {a:np.sqrt(np.diagonal(jack_covs[a])) for a in jack_covs} 
for a in a_list:
    c_params = MLE_params

    fig1 = plt.figure(figsize =(12, 7))

    axs=[fig1.add_axes((0.0,0.4,1,.6)), fig1.add_axes((0.0,0.0,1,.4))]
    plt.subplots_adjust(wspace=0, hspace=0)
    c_data = NvMs[a]

    Ms = M_data[a]
    N = N_data[a]
    edge_pairs = c_data['edge_pairs']

    edges = [edge[0] for edge in edge_pairs]
    edges += [edge_pairs[-1][1]]

    yerr = yerr_dict[a]

    mass_function.set_params(result['x'])

    f_dNdM =  lambda M:mass_function(ccl_cosmo, M/h, a)*vol/(h**3 * M * np.log(10))

    fit_eval = np.array([quad(f_dNdM, edge[0],  edge[1], epsabs=0, epsrel=1e-5)[0] for edge in edge_pairs])

    with open("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/%s_%.2f_NvMfit_output.pkl"%(box, a), "wb") as f:
        pickle.dump({'Ms':Ms, 'tinker_eval':fit_eval, 'N':N, 'edges':edges}, f)

    axs[0].errorbar(Ms, N, yerr, fmt='+', c='black')
    axs[0].scatter(Ms, fit_eval, s=50 , marker='x', c='blue')

    edges = np.array(edges)
    axs[0].bar(x=edges[:-1], height=N, width=np.diff(edges),
               align='edge', fill=False, ec='black', label='Data')
    axs[0].bar(x=edges[:-1], height=fit_eval, width=np.diff(edges), align='edge', fill=False, ec='blue', label='Tinker')
    axs[1].errorbar(Ms, (fit_eval-N)/N, yerr/N, fmt='x', color='blue')

    y1 = 0.1*np.ones_like(N)
    y1 = np.append(y1, y1[-1])
    y1 = np.append(y1[0], y1)

    y2 = -0.1*np.ones_like(N)
    y2 = np.append(y2, y2[-1])
    y2 = np.append(y2[0], y2)

    c_Ms = np.append(Ms, edges[-1])
    c_Ms = np.append(edges[0], c_Ms)
    axs[1].fill_between(c_Ms, y1, y2, alpha=1, color='0.95',label='<10% Error')

    y1 = 0.01*np.ones_like(N)
    y1 = np.append(y1, y1[-1])
    y1 = np.append(y1[0], y1)

    y2 = -0.01*np.ones_like(N)
    y2 = np.append(y2, y2[-1])
    y2 = np.append(y2[0], y2)

    axs[1].fill_between(c_Ms, y1, y2, alpha=1, color='0.85',label='<1% Error')


    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend(frameon=False)
    axs[0].set_ylabel('N')

    axs[1].set_xscale('log')
    # axs[1].set_yscale('lin', linthresh=1e-2)    
    axs[1].legend(frameon=False)
    axs[1].axhline(0, c='black')
    axs[1].set_ylabel('N')
    axs[1].set_xlabel(r'Mass $[h^{-1}M_\odot]$')
    axs[1].set_ylabel(r'$\frac{N_{\rm Tinker}-N_{\rm data}}{N_{\rm data}} $')
    axs[0].set_title('%s, a=%.2f, z=%.2f'%(box, a, scaleToRedshift(a)))
    axs[0].set_ylim(10, 2e5)
    left = np.ceil(np.log10(200*Mpart) * 10) / 10
    axs[0].set_xlim((10**left, np.max(edges)))
    axs[1].set_xlim((10**left, np.max(edges)))
    axs[1].set_ylim((-.29, .29))
    axs[1].set_yticks([-.2, -.1, 0, .1, .2])
    plt.show()
    plt.savefig('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/figures/%s_fit_%.2f.pdf'%(box, a), bbox_inches='tight')



with open("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/%s_params.pkl"%(box), "wb") as f:
    pickle.dump(MLE_params, f)




print('-------------------')
print(MLE_params)
print('Previous: ', prev_params_final)
print('-------------------')

