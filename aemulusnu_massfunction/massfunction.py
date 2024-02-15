from scipy.special import gamma
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os
import emcee
import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.special import gamma
from .utils import *
from classy import Class
from functools import cache, partial


ρcrit0 = 2.77533742639e+11 #h^2 Msol / Mpc^3



from pyccl.halos.halo_model_base import MassFunc



def B(a, M, σM, d, e, f, g):
    oup = e**(d)*g**(-d/2)*gamma(d/2)
    oup += g**(-f/2)*gamma(f/2)
    return 2/oup


def f_G(a, M, σM, d, e, f, g):
    oup = B(a, M, σM, d, e, f, g)
    oup *= ((σM/e)**(-d)+σM**(-f))
    oup *= np.exp(-g/σM**2)
    return oup


def p(p0, p1, a):
    return p0 + (a-0.5)*p1


class MassFuncAemulusNu_fitting_all_snapshot(MassFunc):
    """
    """
    name = 'AemulusNu'

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

    def set_params(self, params):
        self.params = dict(zip(self.params.keys(), params))
        self.paired_params = list(zip(params, params[1:]))[::2]

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        scale_params = dict(zip(['d','e','f','g'],[p(p0, p1, a) for p0, p1 in self.paired_params]))
        return f_G(a, np.exp(lnM), sigM, **scale_params)

    
    
class MassFuncAemulusNu_fitting(MassFunc):
    """
    """
    name = 'AemulusNu'

    def __init__(self, *,
                 mass_def="200m",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "200m"

    def _setup(self):
        self.params = {'d':-1, 'e':-1, 'f':-1, 'g':-1}

    def set_params(self, params):
        self.params = dict(zip(self.params.keys(), params))

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return f_G(a, np.exp(lnM), sigM, **self.params)
