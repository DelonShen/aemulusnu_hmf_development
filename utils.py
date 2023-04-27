import math
from scipy.integrate import quad, fixed_quad
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import functools
from scipy import optimize as optimize
import emcee
from multiprocessing import Pool
from Cosmo import *

ρcrit0 = 2.77533742639e+11 #h^2 Msol / Mpc^3
cosmo_params = pickle.load(open('data/cosmo_params.pkl', 'rb'))

global cosmo

def set_cosmo(cosmo_instance):
    global cosmo
    cosmo = cosmo_instance
    
def scaleToRedshift(a):
    return 1/a-1

def redshiftToScale(z):
    return 1/(1+z)

def dRdM(M, box, a):
    return 1/(6**(2/3)*np.pi**(1/3)*M**(2/3)*cosmo.rhom_a(a)**(1/3))


#Below is a bit ugly, basically for multiprocessing 
#if the data isnt global, emcee pickles the entire 
#dataset every evaluation which slows things down
#significantly. So I am assuming we have a global
#Cosmo object called global to speed things up 400x
#in MCMC chain

def log_prior(param_values):
    #uniform prior
    for a in cosmo.N_data:
        d = param_values[0]
        e = param_values[1]
        f = param_values[2]
        g = param_values[3]

        if(len(param_values)==8):
            d = cosmo.p(a, param_values[0], param_values[1])
            e = cosmo.p(a, param_values[2], param_values[3])
            f = cosmo.p(a, param_values[4], param_values[5])
            g = cosmo.p(a, param_values[6], param_values[7])

        ps = [d,e,f,g]
        for param in ps:
            if(param < 0 or param > 5):
                return -np.inf
    return 0

def log_prob(param_values):   
    """
    Calculates the probability of the given tinker parameters (d, e, f, g)

    Args:
        param_values (np.ndarray): Input array of shape (number of params).

    Returns:
        float: Resulting log probability
    """

    if(log_prior(param_values) == -np.inf):
        return -np.inf

    tinker_fs = {}


    for a in cosmo.N_data:
        params = dict(zip(['d', 'e', 'f', 'g'], param_values))
        
        if(len(param_values)== 8):
            d = cosmo.p(a, param_values[0], param_values[1])
            e = cosmo.p(a, param_values[2], param_values[3])
            f = cosmo.p(a, param_values[4], param_values[5])
            g = cosmo.p(a, param_values[6], param_values[7])
            ps = [d,e,f,g]
            params = dict(zip(['d', 'e', 'f', 'g'], ps))
            
        tinker_eval = [cosmo.tinker(a, M_c,**params)*cosmo.vol for M_c in cosmo.M_numerics]
        f_dndlogM_LOG = interp1d(np.log10(cosmo.M_numerics), tinker_eval, kind='cubic', bounds_error=False, fill_value=0.)
        f_dndlogM = lambda x:f_dndlogM_LOG(np.log10(x))
        tinker_fs[a] = f_dndlogM

    model_vals = {}
    for a in cosmo.N_data:
        model_vals[a] = np.array([quad(tinker_fs[a], edge_pair[0], edge_pair[1], epsabs=1e-1)[0]
            for edge_pair in cosmo.NvMs[a]['edge_pairs']
        ])


    residuals = {a: model_vals[a]-cosmo.N_data[a] for a in model_vals}
    log_probs = [ -0.5 * (np.dot(np.dot(residuals[a].T, cosmo.inv_weighted_cov[a]), residuals[a]) + cosmo.scale_cov[a]) 
                 for a in model_vals]
    if not np.isfinite(np.sum(log_probs)): 
        return -np.inf
    return np.sum(log_probs)

def log_likelihood(param_values):     
    lp = log_prior(param_values)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_prob(param_values)


def fit(param_names = ['d0', 'd1',
                       'e0', 'e1',
                       'f0', 'f1',
                       'g0', 'g1'],
       maxiter=int(8e4)):
    guess = np.random.uniform(size=(len(param_names)))
    while(not np.isfinite(log_likelihood(guess))):
        guess = np.random.uniform(size=(len(param_names)))

    #Start by sampling with a maximum likelihood approach
    nll = lambda *args: -log_likelihood(*args)
    result = optimize.minimize(nll, guess, method="Nelder-Mead", options={
        'maxiter': maxiter
    })
    return result

def fit_MCMC(nwalkers = 32, ndim = 8, n_jumps = 4000, result=None): 
    if(result==None):
        result = fit_individ()
    initialpos = np.array([result['x'] for _ in range(nwalkers)]) + 1e-2 * np.random.normal(size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers = nwalkers,
        ndim = ndim,
        log_prob_fn = log_likelihood,
        pool=Pool()
    )

    sampler.run_mcmc(initialpos, n_jumps, progress=True)
    cosmo.sampler = sampler
    return result, sampler

