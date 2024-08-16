from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.special import gamma
from aemulusnu_hmf import *
# from aemulusnu_hmf_lib.ccl_patches import *

from aemulusnu_hmf.massfunction import *

# import pyccl as ccl
from classy import Class
from functools import cache, partial

from types import MethodType

# from pyccl.halos.halo_model_base import MassFunc
# from pyccl import physical_constants as const






class MassFuncAemulusNu_fitting_all_snapshot():
    """
    """
    name = 'AemulusNu'

    def __init__(self):
        self.params = {'d0':-1, 'd1':-1,
                       'e0':-1, 'e1':-1,
                       'f0':-1, 'f1':-1,
                       'g0':-1, 'g1':-1}
    
        
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
        
        scale_params = dict(zip(['d','e','f','g'],[p(p0, p1, a) for p0, p1 in self.paired_params]))
        
        f = f_G(sigma_cb, **scale_params)
        
        mf = f * rho_cb/M * (-d_ln_sigma_cb_dM)
        return mf
    
    
    def set_params(self, params):
        self.params = dict(zip(self.params.keys(), params))
        self.paired_params = list(zip(params, params[1:]))[::2]
        
        
class MassFuncAemulusNu_fitting_single_snapshot():
    """
    """
    name = 'AemulusNu'

    def __init__(self):
        self.params = {'d':-1,
                       'e':-1,
                       'f':-1,
                       'g':-1,}
    
        
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
                
        f = f_G(sigma_cb, **self.params)
        
        mf = f * rho_cb/M * (-d_ln_sigma_cb_dM)
        return mf
    
    
    def set_params(self, params):
        self.params = dict(zip(self.params.keys(), params))
        self.paired_params = list(zip(params, params[1:]))[::2]


class Tinker08Costanzi13():
    """
    """
    name = 'Tinker08Costanzi13'

   
    def __init__(self):
        # Table 2 of Tinker+08
        delta = np.array(
            [200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.])
        alpha = np.array(
            [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
        beta = np.array(
            [1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        gamma = np.array(
            [2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        phi = np.array(
            [1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
        ldelta = np.log10(delta)
        self.pA0 = interp1d(ldelta, alpha)
        self.pa0 = interp1d(ldelta, beta)
        self.pb0 = interp1d(ldelta, gamma)
        self.pc = interp1d(ldelta, phi)


        
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
        
        f = self._get_fsigma(cosmology, sigma_cb, a, np.log(M))
        
        mf = f * rho_cb/M * (-d_ln_sigma_cb_dM)
        return mf

    def _get_fsigma(self, cosmology, sigM, a, lnM):
        # Eqs (5-8) of Tinker+08
        ld = np.log10(200)
        pA = self.pA0(ld) * a**0.14
        pa = self.pa0(ld) * a**0.06
        pd = 10.**(-(0.75/(ld - 1.8750612633))**1.2)
        pb = self.pb0(ld) * a**pd
        return pA * ((pb / sigM)**pa + 1) * np.exp(-self.pc(ld)/sigM**2)

class Tinker08():
    """
    """
    name = 'Tinker08'

    def __init__(self):
        delta = np.array(
            [200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.])
        alpha = np.array(
            [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
        beta = np.array(
            [1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        gamma = np.array(
            [2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        phi = np.array(
            [1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
        ldelta = np.log10(delta)
        self.pA0 = interp1d(ldelta, alpha)
        self.pa0 = interp1d(ldelta, beta)
        self.pb0 = interp1d(ldelta, gamma)
        self.pc = interp1d(ldelta, phi)

    def __call__(self, cosmology, M, a):
        """ 
        cosmology is a `aemulusnu_hmf_lib.massfunction.cosmology` object
        M is the halo mass in Msol / h
        a is the scale factor
        
        returns the mass function dn/dM in units h^4 / (Mpc^3  Msun)
        """            
        z = scaleToRedshift(a)
        sigma_m = cosmology.sigma_m(M, z)
        d_ln_sigma_m_dM = cosmology.dln_sigma_m_dM(M, z)
        rho_m = cosmology.rho_m_0
        
        f = self._get_fsigma(cosmology, sigma_m, a, np.log(M))
        
        mf = f * rho_m/M * (-d_ln_sigma_m_dM)
        return mf
    
    def _get_fsigma(self, cosmology, sigM, a, lnM):
        ld = np.log10(200)
        pA = self.pA0(ld) * a**0.14
        pa = self.pa0(ld) * a**0.06
        pd = 10.**(-(0.75/(ld - 1.8750612633))**1.2)
        pb = self.pb0(ld) * a**pd
        return pA * ((pb / sigM)**pa + 1) * np.exp(-self.pc(ld)/sigM**2)

