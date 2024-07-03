import math
import pickle
import numpy as np
from types import MethodType

from aemulusnu_mf_lib.utils import *
from aemulusnu_mf_lib.massfunction import *

import torch
import gpytorch


from pyccl.halos.halo_model_base import MassFunc
import pyccl as ccl
from aemulusnu_mf_lib import *
from aemulusnu_mf_lib.ccl_patches import *

from aemulusnu_mf_lib.massfunction import *

from pyccl.halos.halo_model_base import MassFunc
from pyccl import physical_constants as const

from scipy.interpolate import interp1d


import aemulusnu_massfunction
from aemulusnu_massfunction.massfunction_fitting_tinker import *

import os

package_path = os.path.dirname(aemulusnu_massfunction.__file__)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(input_size=train_x.shape[1]), num_tasks=train_y.shape[1]
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
#             gpytorch.kernels.RBFKernel(),
            gpytorch.kernels.SpectralMixtureKernel(num_mixtures=8,ard_num_dims=train_x.shape[1]),
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


class AemulusNu_HMF_Emulator(MassFunc):
    """
    """
    name = 'AemulusNu_HMF_Emulator'

    def __init__(self, *,
                 emulator_loc= package_path+"/emulator.pkl",
                 mass_def="200m",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
        self.ComputedParams = {}
        print('loading emulator from',emulator_loc)
        self.param_names = ['d0', 'd1',
                            'e0', 'e1',
                            'f0', 'f1',
                            'g0', 'g1']
        
        with open(emulator_loc, 'rb') as f:
            self.model, self.in_scaler, self.out_scaler, self.likelihood = pickle.load(f)
            self.model.eval()
            self.likelihood.eval()

    def _get_logM_sigM(self, cosmo, M, a, *, return_dlns=False):
        """Compute ``logM``, ``sigM``, and (optionally) ``dlns_dlogM``.
            Using Costanzi et al. 2013 (JCAP12(2013)012) perscription
            to evaluate HMF in nuCDM cosmology, we replace P_m with P_cb
            """
        if('mirror_cosmo' not in cosmo['extra_parameters']):
            self.init_cosmo(cosmo)

        mirror_cosmo = cosmo['extra_parameters']['mirror_cosmo']
        mirror_cosmo.compute_sigma()  # initialize sigma(M) splines if needed
        logM = np.log10(M)
        # sigma(M)
        status = 0
        sigM, status = lib.sigM_vec(mirror_cosmo.cosmo, a, logM, len(logM), status)
        check(status, cosmo=mirror_cosmo)
        if not return_dlns:
            return logM, sigM

        # dlogsigma(M)/dlog10(M)
        dlns_dlogM, status = lib.dlnsigM_dlogM_vec(mirror_cosmo.cosmo, a, logM,
                                                   len(logM), status)
        check(status, cosmo=mirror_cosmo)
        return logM, sigM, dlns_dlogM

    def init_cosmo(self, cosmo):
        cosmo['extra_parameters']['mirror_cosmo'] = ccl.Cosmology(Omega_c=cosmo['Omega_c'],
                                                 Omega_b=cosmo['Omega_b'],
                                                 h=cosmo['h'],
                                                 A_s=cosmo['A_s'],
                                                 n_s=cosmo['n_s'],
                                                 w0=cosmo['w0'],
                                                 m_nu=cosmo['m_nu'])
        funcType = type(cosmo['extra_parameters']['mirror_cosmo']._compute_linear_power)

        cosmo['extra_parameters']['mirror_cosmo']._compute_linear_power = MethodType(custom_compute_linear_power,
                                                                                     cosmo['extra_parameters']['mirror_cosmo'])
        
    def __call__(self, cosmo, M, a):
        """ Returns the mass function for input parameters. 
            Using Costanzi et al. 2013 (JCAP12(2013)012) perscription
            to evaluate HMF in nuCDM cosmology

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
            M (:obj:`float` or `array`): halo mass.
            a (:obj:`float`): scale factor.

        Returns:
            (:obj:`float` or `array`): mass function \
                :math:`dn/d\\log_{10}M` in units of Mpc^-3 (comoving).
        """
        if('mirror_cosmo' not in cosmo['extra_parameters']):
            self.init_cosmo(cosmo)
            
        M_use = np.atleast_1d(M)
        logM, sigM, dlns_dlogM = self._get_logM_sigM( 
            cosmo, M_use, a, return_dlns=True)

        rho = (const.RHO_CRITICAL
               * (cosmo['Omega_c'] + cosmo['Omega_b'])
               * cosmo['h']**2)

        f = self._get_fsigma(cosmo, sigM, a, 2.302585092994046 * logM)
        mf = f * rho * dlns_dlogM / M_use
        if np.ndim(M) == 0:
            return mf[0]
        return mf


    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "200m"

    def _setup(self):
        self.params = dict(zip(['d0', 'd1',
                            'e0', 'e1',
                            'f0', 'f1',
                            'g0', 'g1'], [-1 for _ in range(8)]))

    def set_params(self, params):
        self.params = dict(zip(self.params.keys(), params))

    def predict_params(self, cosmology, z):
        """
        Parameters:
            - cosmology (dict): A dictioniary containing the cosmological parameters
                - 10^9 As: As * 10^9
                - ns: Spectral index
                - H0: Hubble parameter in [km/s/Mpc]
                - w0: Dark Energy Equation fo State
                - ombh2: Ω_b h^2
                - omch2: Ω_m h^2
                - nu_mass_ev: Neutrino mass sum in [eV]
            - z (float): Redshift to evaluate dn/dM at
        Returns:
            - tinker parameters(dict): A dictionary containing the predicted tinker
                                       parameters from the HMF emulator.
                                       {'d':d, 'e':e, 'f':f, 'g':g}
        """

        a = redshiftToScale(z)


        curr_cosmo_values = get_cosmo_vals(cosmology)
        X = self.in_scaler.transform(np.array([curr_cosmo_values]))
        
        if(tuple(curr_cosmo_values) not in self.ComputedParams):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                 predictions = self.likelihood(self.model(torch.from_numpy(X).float()))
                predictions = self.model(torch.from_numpy(X).float())
                mean = self.out_scaler.inverse_transform(predictions.mean.numpy())
            self.ComputedParams[tuple(curr_cosmo_values)] = dict(zip(self.param_names, mean[0]))
            
        curr_params = list(self.ComputedParams[tuple(curr_cosmo_values)].values())
        paired_params = list(zip(curr_params, curr_params[1:]))[::2]
        
        param_at_z = {'d':-1, 'e':-1, 'f':-1, 'g':-1}
        
        for (p0,p1), key in zip(paired_params, param_at_z):
            param_at_z[key] = p(p0, p1, a)
            
        return param_at_z


    def _get_fsigma(self, cosmo, sigM, a, lnM):
        h = cosmo['h']
        cosmology = {'10^9 As': 10**9 *cosmo['A_s'],
                      'ns': cosmo['n_s'],
                      'H0': cosmo['h']*100,
                      'w0': cosmo['w0'],
                      'ombh2': cosmo['Omega_b']*h**2,
                      'omch2': cosmo['Omega_c']*h**2,
                      'nu_mass_ev': sum(cosmo['m_nu']),}

        cosmo_vals = tuple(get_cosmo_vals(cosmology))


        tinker_params = self.predict_params(cosmology, scaleToRedshift(a))
        return f_G(a, np.exp(lnM), sigM, **tinker_params)




