import numpy as np
from scipy.stats import binned_statistic
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy import optimize as optimize
from multiprocessing import Pool
import pickle
from functools import partial
import functools
from scipy.integrate import quad, fixed_quad
import corner

cosmos_f = open('data/cosmo_params.pkl', 'rb')
cosmo_params = pickle.load(cosmos_f) #cosmo_params is a dict
cosmos_f.close()
ρcrit0 = 2.77533742639e+11 #h^2 Msol / Mpc^3

@functools.cache
def sigma2(pk, R):
    """
    Adapated from https://github.com/komatsu5147/MatterPower.jl
    Computes variance of mass fluctuations with top hat filter of radius R
    For this function let k be the comoving wave number with units h/Mpc

    Parameters:
        - pk (funtion): P(k), the matter power spectrum which has units Mpc^3 / h^3
        - R (float): The smoothing scale in units Mpc/h
    Returns:
        - sigma2 (float): The variance of mass fluctuations
    """

    def dσ2dk(k):
        x = k * R
        W = (3 / x) * (np.sin(x) / x**2 - np.cos(x) / x)
        dσ2dk = W**2 * pk(k) * k**2 / 2 / np.pi**2
        return dσ2dk
    res, err = quad(dσ2dk, 0, np.inf)
    σ2 = res
    return σ2

    
@functools.cache
def dsigma2dR(pk, R):
    """
    Adapated from https://github.com/komatsu5147/MatterPower.jl
    Computes deriative of variance of mass fluctuations wrt top hat filter of radius R
    For this function let k be the comoving wave number with units h/Mpc
    
    Parameters:
        - pk (funtion): P(k), the matter power spectrum which has units Mpc^3 / h^3
        - R (float): The smoothing scale in units Mpc/h
    Returns:
        - dsigma2dR (float): The derivative of the variance of mass fluctuations wrt R
    """

    def dσ2dRdk(k):
        x = k * R
        W = (3 / x) * (np.sin(x) / x**2 - np.cos(x) / x)
        dWdx = (-3 / x) * ((3 / x**2 - 1) * np.sin(x) / x - 3 * np.cos(x) / x**2)
        dσ2dRdk = 2 * W * dWdx * pk(k) * k**3 / 2 / np.pi**2
        return dσ2dRdk
    res, err = quad(dσ2dRdk, 0, np.inf)
    return res

class Cosmo:
    
    def __init__(self, box):
        self.box = box
        
        self.h = cosmo_params[box]['H0']/100

        Pk_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_Pk.pkl'
        Pk_f = open(Pk_fname, 'rb')
        self.Pkz = pickle.load(Pk_f) #Pkz is a dictonary of functions
        Pk_f.close()

        NvM_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_NvsM.pkl'
        NvM_f = open(NvM_fname, 'rb')
        self.NvMs = pickle.load(NvM_f) #NvMs is a dictionary of dictionaries
        NvM_f.close()

        
        jackknife_covs_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/'+box+'_jackknife_covs.pkl'
        jackknife_covs_f = open(jackknife_covs_fname, 'rb')
        jackknife = pickle.load(jackknife_covs_f)
        jackknife_covs_f.close()
        
        
        self.jack_covs = {a:jackknife[a][1] for a in self.NvMs}

        # Compute the weighted covariance matrix incorporating jackknife and poisson
        self.weighted_cov = {a: self.jack_covs[a] for a in self.jack_covs}

        # Inverse of the weighted covariance matrix
        self.inv_weighted_cov = {a:np.linalg.inv(self.weighted_cov[a])  for a in self.weighted_cov}  
        self.scale_cov = {a:np.log(np.linalg.det(self.weighted_cov[a])) for a in self.weighted_cov}

        
        #deal with floating point errors
        self.a_to_z = dict(zip(self.NvMs.keys(), self.Pkz.keys()))
        self.z_to_a = dict(zip(self.Pkz.keys(), self.NvMs.keys()))
        
        
        self.N_data = {}
        self.M_data = {}
        self.aux_data = {}
        
        self.dlnσinvdMs = {}

        self.vol = -1 #Mpc^3/h^3
        self.Mpart = -1
        
        self.M_numerics = -1
        self.sampler = None
        
    def get_scales(self):
        return np.array([a for a in self.a_to_z])
    
    def get_redshifts(self):
        return np.array([z for z in self.z_to_a])
    
    def f_dlnsinvdM(self, f_dlnsinvdlogM_log, M):
        return f_dlnsinvdlogM_log(np.log10(M)) / (M * np.log(10)) 

    def prepare_data(self, a_cur):

        for a in tqdm(a_cur):
            z = self.a_to_z[a]
            Pk = self.Pkz[z]
            c_data = self.NvMs[a]

            Ms = c_data['M'] #units of h^-1 Msolar
            N = c_data['N']
            edge_pairs = c_data['edge_pairs']
            assert(len(Ms) == len(edge_pairs))
            assert(len(Ms) == len(N))


            if(self.vol==-1):
                self.vol = c_data['vol']
            assert(self.vol == c_data['vol'])

            if(self.Mpart==-1):
                self.Mpart = c_data['Mpart']
            assert(self.Mpart == c_data['Mpart'])

            self.N_data[a] = []
            self.M_data[a] = []
            self.aux_data[a] = []
            for N_curr, M_curr, edge_pair in zip(N, Ms, edge_pairs):
                self.N_data[a] += [N_curr]
                self.M_data[a] += [M_curr]
                self.aux_data[a] += [{'a':a, 'edge_pair':edge_pair}]

            self.M_numerics = np.logspace(np.log10(200*self.Mpart), 17, 200) #h^-1 Msolar


            R = [self.M_to_R(m, a) for m in self.M_numerics] #h^-1 Mpc


            M_log10 = np.log10(self.M_numerics)
            sigma2s = [sigma2(Pk, r) for r in R]
            sigma = np.sqrt(sigma2s)
            lnsigmainv = -np.log(sigma)
            dlnsinvdlogM = np.gradient(lnsigmainv, M_log10)

            f_dlnsinvdlogM_log = interp1d(M_log10, dlnsinvdlogM,kind='cubic')
            curr_f_dlnsinvdM = partial(self.f_dlnsinvdM, f_dlnsinvdlogM_log)

            self.dlnσinvdMs[a] = curr_f_dlnsinvdM    
            

    def p(self, a, p0, p1):
        oup = (p0)+(a-0.5)*(p1)
        return oup

    
    def B(self, d, e, f, g):
        oup = e**(d)*g**(-d/2)*gamma(d/2)
        oup += g**(-f/2)*gamma(f/2)
        oup = 2/oup
        return oup

    def f_G(self, a, σM, d, e, f, g):
        oup = self.B(d, e, f, g)
        oup *= ((σM/e)**(-d)+σM**(-f))
        oup *= np.exp(-g/σM**2)
        return oup
    
    def tinker(self, a, M, d, e, f, g):
        R = self.M_to_R(M, a) #Mpc/h
        σM = np.sqrt(sigma2(self.Pkz[self.a_to_z[a]], R))  
        oup = self.f_G(a, σM, d, e, f, g)
        oup *= self.rhom_a(a)/M
        oup *= self.dlnσinvdMs[a](M)
        return oup
    
    @functools.cache
    def M_to_R(self, M, a):
        """
        Converts mass of top-hat filter to radius of top-hat filter

        Parameters:
            - M (float): Mass of the top hat filter in units Msolor/h
            - box (string): Which Aemulus nu box we're considering 
            - a (float): Redshift 

        Returns:
            - R (float): Corresponding radius of top hat filter Mpc/h
        """

        return (M / (4/3 * np.pi * self.rhom_a(a))) ** (1/3) # h^-1 Mpc  
    
    @functools.cache
    def R_to_M(self, R, a):
        """
        Converts radius of top-hat filter to mass of top-hat filter

        Parameters:
            - R (float): Radius of the top hat filter in units Mpc/h
            - box (string): Which Aemulus nu box we're considering 
            - a (float): Redshift 

        Returns:
            - M (float): Corresponding mass of top hat filter Msolar/h 
        """
        return R ** 3 * 4/3 * np.pi * self.rhom_a(a)

    @functools.cache
    def rhom_a(self, a):
        ombh2 = cosmo_params[self.box]['ombh2']
        omch2 = cosmo_params[self.box]['omch2']
        H0 = cosmo_params[self.box]['H0'] #[km s^-1 Mpc-1]
        h = H0/100 

        Ωm = ombh2/h**2 + omch2/h**2

        ΩDE = 1 - Ωm
        wDE = cosmo_params[self.box]['w0'] #'wa' is zero for us

        return Ωm*ρcrit0# TODO ASK *(Ωm*a**(-3) + ΩDE*a**(-3*(1+wDE))) * a**3 # h^2 Msol/Mpc^3
    
    def corner_plot(self, labels = ['d0', 'd1', 'e0', 'e1', 'f0', 'f1', 'g0', 'g1']):
        ndim = len(labels)
        
        N_jumps = len(self.sampler.chain[0])
        samples = self.sampler.chain[:, N_jumps//5*4:, :].reshape((-1, ndim))
        fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],show_titles=True,)
        return fig
    
    def get_MCMC_params(self, labels = ['d0', 'd1', 'e0', 'e1', 'f0', 'f1', 'g0', 'g1']):
        ndim = len(labels)
        
        N_jumps = len(self.sampler.chain[0])
        samples = self.sampler.chain[:, N_jumps//5*4:, :].reshape((-1, ndim))

        final_param_vals = np.percentile(samples,  50,axis=0)
        return dict(zip(labels, final_param_vals))

    def get_MCMC_convergence(self, labels = ['d0', 'd1', 'e0', 'e1', 'f0', 'f1', 'g0', 'g1']):
        ndim = len(labels)
        
        final_params = self.get_MCMC_params(labels=labels)
        final_param_vals = list(final_params.values())
        
        fig, axes = plt.subplots(ndim, figsize=(10, 30), sharex=True)
        samples = self.sampler.get_chain()
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.1)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
#             ax.axhline(result['x'][i], color='red')
            ax.axhline(final_param_vals[i], color='blue')
        axes[-1].set_xlabel("step number");
        return fig, ax
    def get_mass_function_plots(self, yerr_dict, labels = ['d0', 'd1', 'e0', 'e1', 'f0', 'f1', 'g0', 'g1']):
        i=0
        fig_axs = {}
        params_final = self.get_MCMC_params(labels=labels)

        for a in self.N_data:
            z = self.a_to_z[a]
            fig1 = plt.figure(figsize =(12, 7))

            axs=[fig1.add_axes((0.2,0.4,.75,.6)), fig1.add_axes((0.2,0.0,.75,.4))]
            plt.subplots_adjust(wspace=0, hspace=0)
            Pk = self.Pkz[z]
            c_data = self.NvMs[a]

            Ms = self.M_data[a]
            N = self.N_data[a]
            edge_pairs = c_data['edge_pairs']

            edges = [edge[0] for edge in edge_pairs]
            edges += [edge_pairs[-1][1]]

            yerr = yerr_dict[a]
            vol = self.vol
            dM = np.array([edges[1]-edges[0] for edges in edge_pairs])
            dndM = (np.array(N)/vol)/dM
            tinker = self.tinker
            tinker_eval_MCMC = [tinker(a, M_c,**params_final) for M_c in Ms]


            M_numerics = self.M_numerics
            tinker_eval_MCMC = [tinker(a, M_c,**params_final,)*vol for M_c in M_numerics]

            f_dndM_MCMC_LOG = interp1d(np.log10(M_numerics), tinker_eval_MCMC, kind='cubic', bounds_error=False, fill_value=0.)
            f_dndM_MCMC = lambda x:f_dndM_MCMC_LOG(np.log10(x))

            tinker_eval_MCMC = np.array([quad(f_dndM_MCMC, edge[0],  edge[1])[0] for edge in edge_pairs])

            color = plt.colormaps["rainbow"]((i+1)/len(self.Pkz.keys()))[:-1]



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
            axs[0].set_title('%s, a=%.2f, z=%.2f'%(self.box, a, self.a_to_z[a]))
            i+=1

            axs[0].set_xlim((200*self.Mpart, np.max(edges)))
            axs[1].set_xlim((200*self.Mpart, np.max(edges)))
            fig_axs[a] = [fig1, axs]

        #     plt.savefig('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/figures/%s_ML+MCMCFits_a%.2f_individ.pdf'%(box, a), bbox_inches='tight')