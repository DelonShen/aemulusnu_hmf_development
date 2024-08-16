import math
import pickle
import numpy as np
from types import MethodType

from aemulusnu_hmf.utils import *
from aemulusnu_hmf.massfunction import *

import torch
import gpytorch


# from pyccl.halos.halo_model_base import MassFunc
import pyccl as ccl
from aemulusnu_hmf import *
# from aemulusnu_hmf_lib.ccl_patches import *

from aemulusnu_hmf.massfunction import *

# from pyccl.halos.halo_model_base import MassFunc
# from pyccl import physical_constants as const

from scipy.interpolate import interp1d


import aemulusnu_massfunction
from aemulusnu_massfunction.massfunction_fitting_tinker import *

import os
# from pyccl import ccllib as lib

package_path = os.path.dirname(aemulusnu_massfunction.__file__)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(input_size=train_x.shape[1]), num_tasks=train_y.shape[1]
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
#             gpytorch.kernels.SpectralMixtureKernel(num_mixtures=8,ard_num_dims=train_x.shape[1]) + 
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1/2, ard_num_dims=train_x.shape[1]), 
                                         ard_num_dims=train_x.shape[1]) +
            gpytorch.kernels.ConstantKernel(),
            num_tasks=train_y.shape[1], rank=1
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


key_ordering = ['10^9 As', 'ns', 'H0', 'w0', 'ombh2', 'omch2', 'nu_mass_ev']

def get_cosmo_vals(cosmology):
    return [cosmology[curr_key] for curr_key in key_ordering]

def get_cosmo_dict(cosmo_vals):
    return dict(zip(key_ordering, cosmo_vals))


def get_ccl_cosmology(cosmo_vals):
    cosmology = get_cosmo_dict(cosmo_vals)

    h = cosmology['H0']/100
    Ωb =  cosmology['ombh2'] / h**2
    Ωc =  cosmology['omch2'] / h**2

    cosmo = ccl.Cosmology(Omega_c=Ωc,
                          Omega_b=Ωb,
                          h=h,
                          A_s=cosmology['10^9 As']*10**(-9),
                          n_s=cosmology['ns'],
                          w0=cosmology['w0'],
                          m_nu=[cosmology['nu_mass_ev']/3, cosmology['nu_mass_ev']/3, cosmology['nu_mass_ev']/3])

    return cosmo


class MassFuncAemulusNu_GP_emulator_training():
    """
    """
    name = 'AemulusNu'

    def __init__(self, emulator_loc= package_path+"/emulator.pkl"):
        self.params = {'d0':-1, 'd1':-1,
                       'e0':-1, 'e1':-1,
                       'f0':-1, 'f1':-1,
                       'g0':-1, 'g1':-1}

        self.ComputedParams = {}
        self.param_names = ['d0', 'd1',
                            'e0', 'e1',
                            'f0', 'f1',
                            'g0', 'g1']
        with open(emulator_loc, 'rb') as f:
            self.model, self.in_scaler, self.out_scaler, self.likelihood = pickle.load(f)
            self.model.eval()
            self.likelihood.eval()

    def __call__(self, cosmology, M, a):
        """ 
        cosmology is a `aemulusnu_hmf_lib.massfunction.cosmology` object
        M is the halo mass in Msol / h
        a is the scale factor
        
        returns the mass function dn/dM in units h^4 / (Mpc^3  Msun)
        """            
        z = scaleToRedshift(a)
        sigma_cb = cosmology.sigma_cb(M, z)
        d_ln_sigma_cb_dM = cosmology.dln_sigma_cb_dM(M, z)
        rho_cb = cosmology.f_rho_cb(0.0)
        
        tinker_params = self.predict_params(cosmology, scaleToRedshift(a))
        
        f = f_G(sigma_cb, **tinker_params)
        
        mf = f * rho_cb/M * (-d_ln_sigma_cb_dM)
        return mf

    def predict_params(self, cosmology, z):
        """
        Parameters:
            - cosmology `aemulusnu_hmf_lib.massfunction.cosmology` object
            - z (float): Redshift to evaluate dn/dM at
        Returns:
            - tinker parameters(dict): A dictionary containing the predicted tinker
                                       parameters from the HMF emulator.
                                       {'d':d, 'e':e, 'f':f, 'g':g}
        """

        a = redshiftToScale(z)


        curr_cosmo_values = get_cosmo_vals(cosmology.cosmology)
        X = self.in_scaler.transform(np.array([curr_cosmo_values]))
        
        if(tuple(curr_cosmo_values) not in self.ComputedParams):
            with torch.no_grad():#, gpytorch.settings.fast_pred_var():
                predictions = self.model(torch.from_numpy(X).float())
                mean = self.out_scaler.inverse_transform(predictions.mean.numpy())
            self.ComputedParams[tuple(curr_cosmo_values)] = dict(zip(self.param_names, mean[0]))
            
        curr_params = list(self.ComputedParams[tuple(curr_cosmo_values)].values())
        paired_params = list(zip(curr_params, curr_params[1:]))[::2]
        
        param_at_z = {'d':-1, 'e':-1, 'f':-1, 'g':-1}
        
        for (p0,p1), key in zip(paired_params, param_at_z):
            param_at_z[key] = p(p0, p1, a)
            
        return param_at_z

    
    def set_params(self, params):
        self.params = dict(zip(self.params.keys(), params))
        self.paired_params = list(zip(params, params[1:]))[::2]


