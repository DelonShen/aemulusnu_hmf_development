#make cosmo global so emcee doesnt have to pickle all the data making things faster 
global cosmo

from Cosmo import *
import sys
box = sys.argv[1]
cosmo = Cosmo(box)
print(cosmo.h)

from utils import set_cosmo
set_cosmo(cosmo)

from utils import *

import numpy as np
from scipy.stats import binned_statistic
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
import pickle

z_cur = cosmo.get_redshifts()
zlt2 = np.where(z_cur<=2)
z_cur = z_cur[zlt2]
a_cur = np.array([cosmo.z_to_a[z] for z in z_cur])

cosmo.prepare_data(a_cur)

param_names = ['d0', 'd1', 'e0', 'e1', 'f0', 'f1', 'g0', 'g1']
guess = np.random.uniform(size=8)
while(not np.isfinite(log_likelihood(guess))):
    guess = np.random.uniform(size=8)

#Start by sampling with a maximum likelihood approach
nll = lambda *args: -log_likelihood(*args)
result = optimize.minimize(nll, guess, method="Nelder-Mead", options={
    'maxiter': 8*10000
})
result['param_names'] = param_names

print(result)

result_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_MLFit.pkl'
result_f = open(result_fname, 'wb')
pickle.dump(result, result_f)
result_f.close()

result, sampler = fit_MCMC(nwalkers = 64, ndim = 8, n_jumps = 10000, result=result)
with open("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/%s_MCMC_sampler.pkl"%(box), "wb") as f:
    pickle.dump(sampler, f)
    
    
fig = cosmo.corner_plot()
plt.savefig('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/figures/%s_MCMC_corner_1.pdf'%(box), bbox_inches='tight')

cosmo.get_MCMC_convergence()
plt.savefig('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/figures/%s_MCMC_convergence_1.pdf'%(box), bbox_inches='tight')


chain = sampler.get_chain(flat=True, discard=1000)
cov_matrix = np.cov(chain.T)

# compute the eigendecomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# sort the eigenvectors in decreasing order of eigenvalues
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# the eigenvectors form the columns of the orthogonal matrix
cosmo.rotation_matrix = eigenvectors

cosmo.inv_rotation_matrix = cosmo.rotation_matrix.T

with open("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/%s_rotation_matrix.pkl"%(box), "wb") as f:
    pickle.dump(cosmo.inv_rotation_matrix, f)
    
    
N_jumps = len(cosmo.sampler.chain[0])

chain = sampler.get_chain(flat=True, discard=500)
rotated_chain = chain @ cosmo.rotation_matrix

fig = corner.corner(rotated_chain, quantiles=[0.16, 0.5, 0.84],show_titles=True,)
plt.savefig('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/figures/%s_MCMC_corner_1_rotated.pdf'%(box), bbox_inches='tight')


def log_likelihood_rotated(param_values): 
    param_values_unrotated = np.array(param_values) @ cosmo.inv_rotation_matrix
    lp = log_prior(param_values_unrotated)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_prob(param_values_unrotated)



guess = np.random.uniform(size=8)
while(not np.isfinite(log_likelihood_rotated(guess))):
    guess = np.random.uniform(size=8)
print('starting rotated fit')
#Start by sampling with a maximum likelihood approach
nll = lambda *args: -log_likelihood_rotated(*args)
result_rotated = optimize.minimize(nll, guess, method="Nelder-Mead", options={
    'maxiter': 8*10000
})
result_rotated['param_names'] = ['d0_rot', 'd1_rot', 'e0_rot', 'e1_rot', 'f0_rot', 'f1_rot', 'g0_rot', 'g1_rot']

print(result_rotated)

result_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_MLFit_rotated.pkl'
result_f = open(result_fname, 'wb')
pickle.dump(result_rotated, result_f)
result_f.close()


param_values = result_rotated['x']
param_values = np.array(param_values) @ cosmo.inv_rotation_matrix
print(param_values)
i=0
yerr_dict = {a:np.sqrt(np.diagonal(cosmo.weighted_cov[a])) for a in cosmo.weighted_cov}

for a in reversed(cosmo.N_data.keys()):
    z = cosmo.a_to_z[a]
    fig1 = plt.figure(figsize =(12, 7))
    
    d = cosmo.p(a, param_values[0], param_values[1])
    e = cosmo.p(a, param_values[2], param_values[3])
    f = cosmo.p(a, param_values[4], param_values[5])
    g = cosmo.p(a, param_values[6], param_values[7])
    params_final = dict(zip(['d', 'e', 'f', 'g'], [d,e,f,g]))
    axs=[fig1.add_axes((0.2,0.4,.75,.6)), fig1.add_axes((0.2,0.0,.75,.4))]
    plt.subplots_adjust(wspace=0, hspace=0)
    Pk = cosmo.Pkz[z]
    c_data = cosmo.NvMs[a]

    Ms = cosmo.M_data[a]
    N = cosmo.N_data[a]
    edge_pairs = c_data['edge_pairs']

    edges = [edge[0] for edge in edge_pairs]
    edges += [edge_pairs[-1][1]]

    yerr = yerr_dict[a]
    vol = cosmo.vol
    dM = np.array([edges[1]-edges[0] for edges in edge_pairs])
    dndM = (np.array(N)/vol)/dM
    tinker = cosmo.tinker
    tinker_eval_MCMC = [tinker(a, M_c,**params_final) for M_c in Ms]


    M_numerics = cosmo.M_numerics
    tinker_eval_MCMC = [tinker(a, M_c,**params_final,)*vol for M_c in M_numerics]

    f_dndM_MCMC_LOG = interp1d(np.log10(M_numerics), tinker_eval_MCMC, kind='cubic', bounds_error=False, fill_value=0.)
    f_dndM_MCMC = lambda x:f_dndM_MCMC_LOG(np.log10(x))

    tinker_eval_MCMC = np.array([quad(f_dndM_MCMC, edge[0],  edge[1])[0] for edge in edge_pairs])

    color = plt.colormaps["rainbow"]((i+1)/len(cosmo.Pkz.keys()))[:-1]



    axs[0].errorbar(Ms, N, yerr, fmt='+', c='black')
    axs[0].scatter(Ms, tinker_eval_MCMC, s=50 , marker='x', c='blue')

    edges = np.array(edges)
    tmp = 0# edges[:-1]*10**(0.01)-edges[:-1]
    axs[0].bar(x=edges[:-1], height=N, width=np.diff(edges),
               align='edge', fill=False, ec='black', label='Data')
    axs[0].bar(x=edges[:-1]-tmp, height=tinker_eval_MCMC, width=np.diff(edges), align='edge', fill=False, ec='blue', label='Tinker')
    axs[1].errorbar(Ms, (tinker_eval_MCMC-N), yerr, fmt='x', color='blue')

    y1 = 0.1*np.array(N)
    y1 = np.append(y1, y1[-1])
    y1 = np.append(y1[0], y1)

    y2 = -0.1*np.array(N)
    y2 = np.append(y2, y2[-1])
    y2 = np.append(y2[0], y2)

    c_Ms = np.append(Ms, edges[-1])
    c_Ms = np.append(edges[0], c_Ms)
    axs[1].fill_between(c_Ms, y1, y2, alpha=1, color='0.95',label='10% Error')

    y1 = 0.01*np.array(N)
    y1 = np.append(y1, y1[-1])
    y1 = np.append(y1[0], y1)

    y2 = -0.01*np.array(N)
    y2 = np.append(y2, y2[-1])
    y2 = np.append(y2[0], y2)

    axs[1].fill_between(c_Ms, y1, y2, alpha=1, color='0.85',label='1% Error')


    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend(frameon=False)
    axs[0].set_ylabel('N')

    axs[1].set_xscale('log')
    axs[1].set_yscale('symlog', linthresh=1)    
    axs[1].legend(frameon=False)
    axs[1].axhline(0, c='black')
    axs[1].set_ylabel('N')
    axs[1].set_xlabel(r'Mass $[h^{-1}M_\odot]$')
    axs[1].set_ylabel(r'${N_{\rm Tinker}-N_{\rm data}} $')
    axs[0].set_title('%s, a=%.2f, z=%.2f'%(cosmo.box, a, cosmo.a_to_z[a]))
    i+=1

    axs[0].set_xlim((200*cosmo.Mpart, np.max(edges)))
    axs[1].set_xlim((200*cosmo.Mpart, np.max(edges)))

    plt.savefig('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/figures/ROTATED_%s_MLFits_a%.2f.pdf'%(box, a), bbox_inches='tight')
    
    
    
nwalkers = 64
initialpos = np.array([result_rotated['x'] for _ in range(nwalkers)]) + 1e-2 * np.random.normal(size=(nwalkers, 8))
sampler_rotated = emcee.EnsembleSampler(
    nwalkers = nwalkers,
    ndim = 8,
    log_prob_fn = log_likelihood_rotated,
    pool=Pool()
)

sampler_rotated.run_mcmc(initialpos, 10000, progress=True);

with open("/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/%s_MCMC_sampler_rotated.pkl"%(box), "wb") as f:
    pickle.dump(sampler_rotated, f)
    
    
    
fig, axes = plt.subplots(8, figsize=(10, 30), sharex=True)
samples = sampler_rotated.get_chain()
for i in range(8):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.1)
    ax.set_xlim(0, len(samples))
#     ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
#     ax.axhline(final_param_vals[i], color='blue')
axes[-1].set_xlabel("step number");
plt.savefig('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/figures/%s_MCMC_convergence_rotated.pdf'%(box), bbox_inches='tight')



N_jumps = len(sampler_rotated.chain[0])
samples = sampler_rotated.chain[:, 5000:, :].reshape((-1, 8))
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84],show_titles=True,)
plt.savefig('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/figures/%s_MCMC_corner_2_rotated.pdf'%(box), bbox_inches='tight')



N_jumps = len(sampler_rotated.chain[0])
samples = sampler_rotated.chain[:, N_jumps//5*4:, :].reshape((-1, 8))

final_param_vals = np.percentile(samples,  50,axis=0)

param_values = final_param_vals
param_values = np.array(param_values) @ cosmo.inv_rotation_matrix

# param_values = result['x']
i=0
yerr_dict = {a:np.sqrt(np.diagonal(cosmo.weighted_cov[a])) for a in cosmo.weighted_cov}

for a in reversed(cosmo.N_data.keys()):
    z = cosmo.a_to_z[a]
    fig1 = plt.figure(figsize =(12, 7))
    
    d = cosmo.p(a, param_values[0], param_values[1])
    e = cosmo.p(a, param_values[2], param_values[3])
    f = cosmo.p(a, param_values[4], param_values[5])
    g = cosmo.p(a, param_values[6], param_values[7])
    params_final = dict(zip(['d', 'e', 'f', 'g'], [d,e,f,g]))
    axs=[fig1.add_axes((0.2,0.4,.75,.6)), fig1.add_axes((0.2,0.0,.75,.4))]
    plt.subplots_adjust(wspace=0, hspace=0)
    Pk = cosmo.Pkz[z]
    c_data = cosmo.NvMs[a]

    Ms = cosmo.M_data[a]
    N = cosmo.N_data[a]
    edge_pairs = c_data['edge_pairs']

    edges = [edge[0] for edge in edge_pairs]
    edges += [edge_pairs[-1][1]]

    yerr = yerr_dict[a]
    vol = cosmo.vol
    dM = np.array([edges[1]-edges[0] for edges in edge_pairs])
    dndM = (np.array(N)/vol)/dM
    tinker = cosmo.tinker
    tinker_eval_MCMC = [tinker(a, M_c,**params_final) for M_c in Ms]


    M_numerics = cosmo.M_numerics
    tinker_eval_MCMC = [tinker(a, M_c,**params_final,)*vol for M_c in M_numerics]

    f_dndM_MCMC_LOG = interp1d(np.log10(M_numerics), tinker_eval_MCMC, kind='cubic', bounds_error=False, fill_value=0.)
    f_dndM_MCMC = lambda x:f_dndM_MCMC_LOG(np.log10(x))

    tinker_eval_MCMC = np.array([quad(f_dndM_MCMC, edge[0],  edge[1])[0] for edge in edge_pairs])

    color = plt.colormaps["rainbow"]((i+1)/len(cosmo.Pkz.keys()))[:-1]



    axs[0].errorbar(Ms, N, yerr, fmt='+', c='black')
    axs[0].scatter(Ms, tinker_eval_MCMC, s=50 , marker='x', c='blue')

    edges = np.array(edges)
    tmp = 0# edges[:-1]*10**(0.01)-edges[:-1]
    axs[0].bar(x=edges[:-1], height=N, width=np.diff(edges),
               align='edge', fill=False, ec='black', label='Data')
    axs[0].bar(x=edges[:-1]-tmp, height=tinker_eval_MCMC, width=np.diff(edges), align='edge', fill=False, ec='blue', label='Tinker')
    axs[1].errorbar(Ms, (tinker_eval_MCMC-N), yerr, fmt='x', color='blue')

    y1 = 0.1*np.array(N)
    y1 = np.append(y1, y1[-1])
    y1 = np.append(y1[0], y1)

    y2 = -0.1*np.array(N)
    y2 = np.append(y2, y2[-1])
    y2 = np.append(y2[0], y2)

    c_Ms = np.append(Ms, edges[-1])
    c_Ms = np.append(edges[0], c_Ms)
    axs[1].fill_between(c_Ms, y1, y2, alpha=1, color='0.95',label='10% Error')

    y1 = 0.01*np.array(N)
    y1 = np.append(y1, y1[-1])
    y1 = np.append(y1[0], y1)

    y2 = -0.01*np.array(N)
    y2 = np.append(y2, y2[-1])
    y2 = np.append(y2[0], y2)

    axs[1].fill_between(c_Ms, y1, y2, alpha=1, color='0.85',label='1% Error')


    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend(frameon=False)
    axs[0].set_ylabel('N')

    axs[1].set_xscale('log')
    axs[1].set_yscale('symlog', linthresh=1)    
    axs[1].legend(frameon=False)
    axs[1].axhline(0, c='black')
    axs[1].set_ylabel('N')
    axs[1].set_xlabel(r'Mass $[h^{-1}M_\odot]$')
    axs[1].set_ylabel(r'${N_{\rm Tinker}-N_{\rm data}} $')
    axs[0].set_title('%s, a=%.2f, z=%.2f'%(cosmo.box, a, cosmo.a_to_z[a]))
    i+=1

    axs[0].set_xlim((200*cosmo.Mpart, np.max(edges)))
    axs[1].set_xlim((200*cosmo.Mpart, np.max(edges)))

    plt.savefig('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/figures/ROTATED_%s_ML+MCMCFits_a%.2f.pdf'%(box, a), bbox_inches='tight')
