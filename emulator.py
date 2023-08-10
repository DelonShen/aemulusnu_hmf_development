import math
import pickle
import numpy as np

from utils import *
from massfunction import *

import torch
import gpytorch


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(input_size=train_x.shape[1]), num_tasks=train_y.shape[1]
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
             gpytorch.kernels.SpectralMixtureKernel(num_mixtures=3,
                                                    ard_num_dims=train_x.shape[1]),
            num_tasks=train_y.shape[1], rank=1
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class AemulusNu_HMF_Emulator:
    """
    Halo Mass Function Emulator,
    built from Aemulus-ν suite of simulations
    """
    def __init__(self):
        #TODO eventaully replace with local emuatlor
        self.MassFunctions = {}
        self.ComputedParams = {}
#         with open("GP_emulator.pkl", "rb") as f:
        with open('/scratch/users/delon/aemulusnu_massfunction/GP_loBox0_1400.pkl', 'rb') as f:
            self.model, self.in_scaler, self.out_scaler, self.likelihood = pickle.load(f)
            self.model.eval()
            self.likelihood.eval()

    def get_massfunction(self, cosmology):
        cosmology['As'] = cosmology['10^9 As'] #account for poor naming convention in MassFunction

        curr_cosmo_values = self.get_cosmo_vals(cosmology)

        if(tuple(curr_cosmo_values) not in self.MassFunctions):
            self.MassFunctions[tuple(curr_cosmo_values)] = MassFunction(cosmology)

        return self.MassFunctions[tuple(curr_cosmo_values)]

    def get_cosmo_vals(self, cosmology):
        key_ordering = ['10^9 As', 'ns', 'H0', 'w0', 'ombh2', 'omch2', 'nu_mass_ev', 'sigma8']
        return [cosmology[curr_key] for curr_key in key_ordering]


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
                - sigma8: σ8
            - z (float): Redshift to evaluate dn/dM at
        Returns:
            - tinker parameters(dict): A dictionary containing the predicted tinker
                                       parameters from the HMF emulator.
                                       {'d':d, 'e':e, 'f':f, 'g':g}
        """

        a = redshiftToScale(z)

        mass_function = self.get_massfunction(cosmology)
        mass_function.compute_dlnsinvdM(a)

        sigma8 = mass_function.pkclass.sigma(8, z, h_units=True) #sigma8 at current redshift

        curr_cosmo_values = self.get_cosmo_vals(cosmology)
        X = self.in_scaler.transform(np.array([curr_cosmo_values + [a, sigma8]]))
        if(tuple(X[0].tolist()) in self.ComputedParams):
            return self.ComputedParams[tuple(X[0].tolist())]

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(torch.from_numpy(X).float()))
            mean = self.out_scaler.inverse_transform(predictions.mean.numpy())
        self.ComputedParams[tuple(X[0].tolist())] = dict(zip(['d','e','f','g'], mean[0]))
        return self.ComputedParams[tuple(X[0].tolist())]


    def predict_dndm(self, cosmology, z, m):
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
                - sigma8: σ8
            - z (float): Redshift to evaluate dn/dM at
            - m (float): Mass [M_solar / h] to evaluate dn/dM at
        Returns:
            - dn/dm(m,z) (float): Halo Mass Function evaluated at mass 'm'
                           during redshift 'z' for given cosmology
                           [h^4 Mpc^-3 Msolar^(-1)]
        """
        a = redshiftToScale(z)

        tinker_params = self.predict_params(cosmology, z)

        mass_function = self.get_massfunction(cosmology)
        mass_function.compute_dlnsinvdM(a)

        return mass_function.tinker(a, m, **tinker_params)

    def predicit_n_in_bins(self, cosmology, z, bin_edges):
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
                - sigma8: σ8
            - z (float): Redshift to evaluate dn/dM at
            - bins_edges (list): List of Mass [M_solar / h] bin edges to evaluate dn/dM in
        Returns:
            - n_in_bins (list): List of number density in mass bins

        """
        return [-1]
