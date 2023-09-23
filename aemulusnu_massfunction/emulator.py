import math
import pickle
import numpy as np

from .utils import *
from .massfunction import *

import torch
import gpytorch


from pyccl.halos.halo_model_base import MassFunc
import pyccl as ccl

from scipy.interpolate import interp1d



class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(input_size=train_x.shape[1]), num_tasks=train_y.shape[1]
        )
        print(1)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            #gpytorch.kernels.RBFKernel(),
            gpytorch.kernels.SpectralMixtureKernel(num_mixtures=3,ard_num_dims=train_x.shape[1]),
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


@cache
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
                 emulator_loc= '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/GP_loBox0_1400.pkl',
                 mass_def="200m",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
        self.ComputedParams = {}
        with open(emulator_loc, 'rb') as f:
            self.model, self.in_scaler, self.out_scaler, self.likelihood = pickle.load(f)
            self.model.eval()
            self.likelihood.eval()



    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "fof"

    def _setup(self):
        self.params = {'d':-1, 'e':-1, 'f':-1, 'g':-1}

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
        X = self.in_scaler.transform(np.array([curr_cosmo_values + [a]]))
        if(tuple(X[0].tolist()) in self.ComputedParams):
            return self.ComputedParams[tuple(X[0].tolist())]

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(torch.from_numpy(X).float()))
            mean = self.out_scaler.inverse_transform(predictions.mean.numpy())
        self.ComputedParams[tuple(X[0].tolist())] = dict(zip(['d','e','f','g'], mean[0]))
        return self.ComputedParams[tuple(X[0].tolist())]


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
