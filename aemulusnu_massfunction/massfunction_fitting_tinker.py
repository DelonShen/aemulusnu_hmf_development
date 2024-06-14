from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.special import gamma
from aemulusnu_mf_lib import *
from aemulusnu_mf_lib.ccl_patches import *

from aemulusnu_mf_lib.massfunction import *

import pyccl as ccl
from classy import Class
from functools import cache, partial

from types import MethodType

from pyccl.halos.halo_model_base import MassFunc
from pyccl import physical_constants as const






class MassFuncAemulusNu_fitting_all_snapshot(MassFunc):
    """
    """
    name = 'AemulusNu'
    mirror_cosmos = {}

    def __init__(self, *,
                 mass_def="200m",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "200m"

    def _setup(self):
        self.params = {'d0':-1, 'd1':-1,
                       'e0':-1, 'e1':-1,
                       'f0':-1, 'f1':-1,
                       'g0':-1, 'g1':-1}

    
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
    
    
    def set_params(self, params):
        self.params = dict(zip(self.params.keys(), params))
        self.paired_params = list(zip(params, params[1:]))[::2]

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        scale_params = dict(zip(['d','e','f','g'],[p(p0, p1, a) for p0, p1 in self.paired_params]))
        return f_G(a, np.exp(lnM), sigM, **scale_params)