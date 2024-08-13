from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline
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
        
        f = f_G(a, M, sigma_cb, **scale_params)
        
        mf = f * rho_cb/M * (-d_ln_sigma_cb_dM)
        return mf
    
    
    def set_params(self, params):
        self.params = dict(zip(self.params.keys(), params))
        self.paired_params = list(zip(params, params[1:]))[::2]