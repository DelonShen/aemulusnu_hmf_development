import sys
import time

import sys

import time

import sys



import numpy as np
from scipy import optimize as optimize
from aemulusnu_massfunction.utils import *
from aemulusnu_massfunction.massfunction import *

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
import pickle
import pyccl as ccl
from scipy.stats import poisson


import numpy as np
from scipy import optimize as optimize

import numpy as np
from tqdm import tqdm, trange
import os
import numpy as np
import pickle


a_list_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/alist.pkl'
a_list_f = open(a_list_fname, 'rb')
a_list = pickle.load(a_list_f) 
a_list_f.close()
print('alist', a_list)

import subprocess
from datetime import date


box_prev =  sys.argv[2]
box = sys.argv[1]
print('Curr: %-10s, Prev: %-10s'%(box, box_prev))



def do(box, a_fit, prev_box, prev_a):
  

    print('##############################')
    print()
    print()

#     KX = np.diag([1e-1, 1e-4, 1e-3, 1e-5])
#     KX = np.diag([1e-1, 1e-4, 1e-3, 5e-4])

#     _curr_z = scaleToRedshift(a_fit)
#     _prev_z = scaleToRedshift(prev_a)
#     _delta_z = _curr_z - _prev_z
#     KX = np.diag([1e-1, 5e-2, 1e-4, 1e-4])#*(1+_delta_z)
    
    
    
    KX = np.diag([1e-2, 1e-4, 1e-4, 1e-4])


    
    
    
    assert_g_increasing = not np.isclose(a_fit, prev_a)
    if(assert_g_increasing):
        print('asserting exponential supression start at lower masses')


    # KX = np.diag([1e-5, 1e-5, 1e-6, 1e-3])

    # if(a_fit == 1.0):
    #     KX = np.diag([1e-5, 1e-5, 1e-6, 1e-5])

    param_names = ['d','e','f','g']
    ndim = len(param_names)


    cosmos_f = open('../data/cosmo_params.pkl', 'rb')
    cosmo_params = pickle.load(cosmos_f) #cosmo_params is a dict
    cosmos_f.close()



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



    cosmo = cosmo_params[box]

    mass_function = MassFuncAemulusNu_fitting()

    h = cosmo['H0']/100

    NvM_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_NvsM.pkl'
    NvM_f = open(NvM_fname, 'rb')
    NvMs = pickle.load(NvM_f) #NvMs is a dictionary of dictionaries
    NvM_f.close()

    all_as = list(NvMs.keys())
    all_zs = list(map(scaleToRedshift, all_as))


    N_data = {}
    M_data = {}
    aux_data = {}


    vol = -1 #Mpc^3/h^3
    Mpart = -1

    if(a_fit not in all_as):
        print('not enough data to fit this snapshot', box, a_fit)
        sys.exit()

    for a in [a_fit]:
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

    from scipy.stats import poisson


    M_numerics = np.logspace(np.log10(100*Mpart), 16, 50) #Msol/h

    jackknife_covs_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_jackknife_covs.pkl'
    jackknife_covs_f = open(jackknife_covs_fname, 'rb')
    jackknife = pickle.load(jackknife_covs_f)
    jackknife_covs_f.close()

    jack_covs = {a:jackknife[a][1] for a in N_data}

    weighted_cov = {a: jack_covs[a] for a in jack_covs}

    inv_weighted_cov = {a:np.linalg.inv(weighted_cov[a]) for a in weighted_cov}  

    scale_cov = {a:np.log(np.linalg.det(weighted_cov[a])) for a in weighted_cov}


    def uniform_log_prior(param_values):
        #uniform prior
        for param in param_values:
            if(param < 0 or param > 5):
                return -np.inf
        return 0


    def log_prob(param_values):   
        """
        Calculates the probability of the given tinker parameters 

        Args:
            param_values (np.ndarray): Input array of shape (number of params).

        Returns:
            float: Resulting log probability
        """

        if(uniform_log_prior(param_values) == -np.inf):
            return -np.inf

        params = dict(zip(param_names, param_values))

        tinker_fs = {}

        mass_function.set_params(param_values)
    #     tinker_eval = mass_function(ccl_cosmo, M_numerics/h, a_fit)*vol/(h**3 * M_numerics * np.log(10))
        f_dNdM = lambda M:mass_function(ccl_cosmo, M/h, a_fit)*vol/(h**3 * M * np.log(10))
        tinker_fs[a_fit] = f_dNdM

        model_vals = {}
        model_vals[a_fit] = np.array([quad(tinker_fs[a_fit], edge_pair[0], edge_pair[1], epsabs=0, epsrel=5e-3)[0]
            for edge_pair in NvMs[a_fit]['edge_pairs']
        ])


        residuals = {a: model_vals[a]-N_data[a] for a in model_vals}
        log_probs = [ -0.5 * (len(inv_weighted_cov)* np.log(2*np.pi) + 
                              np.dot(np.dot(residuals[a].T, inv_weighted_cov[a]), residuals[a]) + 
                              scale_cov[a]) 
                     for a in model_vals]
        if not np.isfinite(np.sum(log_probs)): 
            return -np.inf
        return np.sum(log_probs)

    def log_likelihood(param_values):
        lp = uniform_log_prior(param_values)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_prob(param_values)





    prev_params_final = None
    with open("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/%s_%.2f_params.pkl"%(prev_box, prev_a), "rb") as f:
        prev_params_final = pickle.load(f)
    prev_final_param_vals = np.array(list(prev_params_final.values()))
    print('prev', prev_params_final)
    KXinv = np.linalg.inv(KX)
    detKX = np.linalg.det(KX)
    logdetKX = np.log(detKX)

    def log_prior(param_values):
        xmp = param_values - prev_final_param_vals
        arg = np.dot(np.dot(xmp.T, KXinv), xmp)
        if(assert_g_increasing and param_values[-1] < prev_final_param_vals[-1]): #assert exponnteitla supression increases as redshift increases
            return -np.inf

        return -1/2*(ndim * np.log(2*np.pi) + logdetKX + arg)

    def log_likelihood_with_prior(param_values):
        lp = log_prior(param_values)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_prob(param_values)


    guess = prev_final_param_vals
    print('Starting ML Fit')
    #Start by sampling with a maximum likelihood approach

    bounds = [(0,5) for _ in range(len(guess))]
    nll = lambda *args: -log_likelihood_with_prior(*args)
    # import cProfile
    # cProfile.run("optimize.minimize(nll, guess, method='Nelder-Mead', bounds = bounds, options={'maxiter': len(guess)*10000})")

    result = optimize.minimize(nll, guess,  bounds = bounds, method='Nelder-Mead', options={
        'maxiter': len(guess)*10000
    })
    result['param_names'] = param_names
    print(box)
    print(result)
    print(result['x'])


    MLE_params = dict(zip(param_names, result['x']))
    print(MLE_params)
    print('Previous: ', prev_params_final)

    with open("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/%s_%.2f_params.pkl"%(box, a_fit), "wb") as f:
        pickle.dump(MLE_params, f)

    yerr_dict = {a:np.sqrt(np.diagonal(weighted_cov[a])) for a in weighted_cov}
    c_params = MLE_params
    a = a_fit

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
    dM = np.array([edges[1]-edges[0] for edges in edge_pairs])



    # tinker_evaled = mass_function(ccl_cosmo, M_numerics/h, a_fit)*vol/(h**3 * M_numerics * np.log(10))
    f_dNdM =  lambda M:mass_function(ccl_cosmo, M/h, a_fit)*vol/(h**3 * M * np.log(10))

    tinker_eval_MCMC = np.array([quad(f_dNdM, edge[0],  edge[1], epsabs=0, epsrel=1e-5)[0] for edge in edge_pairs])



    axs[0].errorbar(Ms, N, yerr, fmt='+', c='black')
    axs[0].scatter(Ms, tinker_eval_MCMC, s=50 , marker='x', c='blue')

    edges = np.array(edges)
    axs[0].bar(x=edges[:-1], height=N, width=np.diff(edges),
               align='edge', fill=False, ec='black', label='Data')
    axs[0].bar(x=edges[:-1], height=tinker_eval_MCMC, width=np.diff(edges), align='edge', fill=False, ec='blue', label='Tinker')
    axs[1].errorbar(Ms, (tinker_eval_MCMC-N)/N, yerr/N, fmt='x', color='blue')

    with open("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/%s_%.2f_NvMfit_output.pkl"%(box, a_fit), "wb") as f:
        pickle.dump({'Ms':Ms, 'tinker_eval':tinker_eval_MCMC, 'N':N, 'edges':edges}, f)

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

    left = np.ceil(np.log10(200*Mpart) * 10) / 10
    axs[0].set_xlim((10**left, np.max(edges)))
    axs[1].set_xlim((10**left, np.max(edges)))
    axs[1].set_ylim((-.29, .29))
    axs[1].set_yticks([-.2, -.1, 0, .1, .2])
    plt.savefig('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/figures/%s_fit_%.2f.pdf'%(box, a), bbox_inches='tight')

    print()
    print()
# line = 'python -u fit_individ_iter.py %s %f %s %f'%(box, 1.0, box_prev, 1.0)
# print(line)
# result = subprocess.run(line, shell=True, check=True)
do(box, 1.0, box_prev, 1.0)
for i in trange(1, len(a_list)):
    do(box, a_list[i], box, a_list[i-1])
#     line = 'python -u fit_individ_iter.py %s %f %s %f'%(box, a_list[i], box, a_list[i-1])
#     print(line)
#     result = subprocess.run(line, shell=True, check=True)


