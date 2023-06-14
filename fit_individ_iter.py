import sys 
box = sys.argv[1]
a_fit = eval(sys.argv[2])
prev_box = sys.argv[3]
prev_a = eval(sys.argv[4])
PARAM_SPREAD = 1e-4

param_names = ['d','e','f','g']
ndim = len(param_names)


from utils import *
from massfunction import *

import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
import pickle

cosmos_f = open('data/cosmo_params.pkl', 'rb')
cosmo_params = pickle.load(cosmos_f) #cosmo_params is a dict
cosmos_f.close()


cosmo = cosmo_params[box]
mass_function = MassFunction(cosmo)

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
from scipy.interpolate import interp1d, UnivariateSpline, InterpolatedUnivariateSpline


vol = -1 #Mpc^3/h^3
Mpart = -1

for a in tqdm([a_fit]):
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
    
    mass_function.compute_dlnsinvdM(a)
    
    
a_list = list(NvMs.keys())

from scipy.stats import poisson


M_numerics = np.logspace(np.log10(100*Mpart), 17, 50)

jackknife_covs_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_jackknife_covs.pkl'
jackknife_covs_f = open(jackknife_covs_fname, 'rb')
jackknife = pickle.load(jackknife_covs_f)
jackknife_covs_f.close()

jack_covs = {a:jackknife[a][1] for a in N_data}

# Compute the weighted covariance matrix incorporating jackknife and poisson
weighted_cov = {a: jack_covs[a] for a in jack_covs}

# Inverse of the weighted covariance matrix
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
    
    for a in N_data:
        tinker_eval = [mass_function.tinker(a, M_c, **params)*vol for M_c in M_numerics]
        f_dndlogM = interp1d(M_numerics, tinker_eval, kind='linear', bounds_error=False, fill_value=0.)
        tinker_fs[a] = f_dndlogM
        
    model_vals = {}
    for a in N_data:
        if(scaleToRedshift(a) >=2):
            continue
        model_vals[a] = np.array([quad(tinker_fs[a], edge_pair[0], edge_pair[1], epsabs=1e-1)[0]
            for edge_pair in NvMs[a]['edge_pairs']
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
KX = np.diag([PARAM_SPREAD for _ in range(ndim)])
KXinv = np.linalg.inv(KX)
detKX = np.linalg.det(KX)
logdetKX = np.log(detKX)

def log_prior(param_values):
    
    xmp = param_values - prev_final_param_vals
    arg = np.dot(np.dot(xmp.T, KXinv), xmp)
    return -1/2*(ndim * np.log(2*np.pi) + logdetKX + arg)
def log_likelihood_with_prior(param_values):
    lp = log_prior(param_values)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_prob(param_values)


guess = prev_final_param_vals + np.random.normal(loc=0, 
                                                 scale=PARAM_SPREAD, 
                                                 size=prev_final_param_vals.shape)    
print('Starting ML Fit')
#Start by sampling with a maximum likelihood approach
from scipy import optimize as optimize
nll = lambda *args: -log_likelihood_with_prior(*args)
result = optimize.minimize(nll, guess, method="Nelder-Mead", bounds = [(0,10) for _ in range(ndim)], options={
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
Pk = mass_function.Pka[a]
c_data = NvMs[a]

Ms = M_data[a]
N = N_data[a]
edge_pairs = c_data['edge_pairs']

edges = [edge[0] for edge in edge_pairs]
edges += [edge_pairs[-1][1]]

yerr = yerr_dict[a]
dM = np.array([edges[1]-edges[0] for edges in edge_pairs])


tinker_eval_MCMC = [mass_function.tinker(a, M_c, **c_params)*vol for M_c in M_numerics]
f_dndM_MCMC =  interp1d(M_numerics, tinker_eval_MCMC, kind='linear', bounds_error=False, fill_value=0.)

tinker_eval_MCMC = np.array([quad(f_dndM_MCMC, edge[0],  edge[1])[0] for edge in edge_pairs])



axs[0].errorbar(Ms, N, yerr, fmt='+', c='black')
axs[0].scatter(Ms, tinker_eval_MCMC, s=50 , marker='x', c='blue')

edges = np.array(edges)
axs[0].bar(x=edges[:-1], height=N, width=np.diff(edges),
           align='edge', fill=False, ec='black', label='Data')
axs[0].bar(x=edges[:-1], height=tinker_eval_MCMC, width=np.diff(edges), align='edge', fill=False, ec='blue', label='Tinker')
axs[1].errorbar(Ms, (tinker_eval_MCMC-N)/N, yerr/N, fmt='x', color='blue')

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