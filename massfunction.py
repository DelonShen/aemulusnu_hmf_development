from scipy.special import gamma
from scipy.optimize import curve_fit
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
import dill as pickle
from functools import partial
import functools
from scipy.integrate import quad, fixed_quad
import corner
from utils import *
from classy import Class

ρcrit0 = 2.77533742639e+11 #h^2 Msol / Mpc^3


class MassFunction:
    def __init__(self, cosmology, fixed={}):
        '''
        TODO: note that currently cosmo['As'] is actually 10^9 A_s,
        fix sometime

        Comoving halo mass function

        '''
        self.cosmology = cosmology
        self.fixed = fixed

        cosmo = self.cosmology
        h = cosmo['H0']/100
        cosmo_dict = {
            'h': h,
            'Omega_b': cosmo['ombh2'] / h**2,
            'Omega_cdm': cosmo['omch2'] / h**2,
            'N_ur': 0.00641,
            'N_ncdm': 1,
            'output': 'mPk mTk',
            'z_pk': '0.0,99',
            'P_k_max_h/Mpc': 20.,
            'm_ncdm': cosmo['nu_mass_ev']/3,
            'deg_ncdm': 3,
            'T_cmb': 2.7255,
            'A_s': cosmo['As'] * 10**-9,
            'n_s': cosmo['ns'],
            'Omega_Lambda': 0.0,
            'w0_fld': cosmo['w0'],
            'wa_fld': 0.0,
            'cs2_fld': 1.0,
            'fluid_equation_of_state': "CLP"
        }
        self.pkclass = Class()
        self.pkclass.set(cosmo_dict)
        self.pkclass.compute()

        N_snapshots = 16
        self.Pka = {}
        self.dlnσinvdMs = {}

    def compute_dlnsinvdM(self, a):
        h = self.cosmology['H0']/100
        M_numerics = np.logspace(11, 17, 100) #h^-1 Msolar
        R = self.M_to_R(M_numerics, a) #h^-1 Mpc

        if(a not in self.Pka):
            self.compute_Pka(a)
        if(a in self.dlnσinvdMs):
            return

        Pk = self.Pka[a]
        sigma = [self.pkclass.sigma(R_curr, scaleToRedshift(a)) for R_curr in R]
        sigma2s = np.square(sigma)

        ds2dR = dsigma2dR(Pk, R)
        dRdMs = self.dRdM(M_numerics, a)
        ds2dM = ds2dR * dRdMs

        dlnsinvds2 = -1/(2*sigma2s)
        dlnsinvdM = ds2dM*dlnsinvds2

        f_dlnsinvdM_log = interp1d(np.log10(M_numerics), dlnsinvdM, kind='linear')
        self.dlnσinvdMs[a] = lambda x:f_dlnsinvdM_log(np.log10(x))

    def compute_Pka(self, a):
        h = self.cosmology['H0']/100
        z = scaleToRedshift(a)

        kt = np.logspace(-3, 1, 100) # h/Mpc
        pk_m_lin = np.array(
            [
                self.pkclass.pk_lin(ki, np.array([z]))*h**3 #units of Mpc^3/h^3
                for ki in kt * h # 1 / Mpc
            ]
        )
        from scipy.interpolate import interp1d
        #given k in units of h/Mpc gives Pk in units of Mpc^3/h^3 
        Pk = interp1d(kt, pk_m_lin, kind='linear', bounds_error=False, fill_value=0.)
        self.Pka[a] = Pk
#        class_sigma8 = self.pkclass.sigma(8, z, h_units=True)
#        my_sigma8 = np.sqrt(sigma2(Pk, 8)) # 8 h^-1 Mpc
#        assert(np.abs(class_sigma8-my_sigma8)<0.01*class_sigma8)

    def tinker(self, a, M, d, e, f, g):
        R = self.M_to_R(M, a) #Mpc/h
        σM = self.pkclass.sigma(R, scaleToRedshift(a)) # unitelss
        oup = self.f_G(a, M, σM, d, e, f, g) #unitless
        oup *= self.rhom_a(a)/M # h^3 /Mpc^3
        oup *= self.dlnσinvdMs[a](M) # h / Msun
        return oup # h^4 / (Mpc^3  Msun)
    
    def tinker_wrapper(self, a, M, params):
        d = self.p(a, params['d0'], params['d1'])
        e = self.p(a, params['e0'], params['e1'])
        f = self.p(a, params['f0'], params['f1'])
        g = self.p(a, params['g0'], params['g1'])
        return self.tinker(a, M, d, e, f, g)

    def p(self, a, p0, p1):
        oup = (p0)+(a-0.5)*(p1)
        return oup

    def B(self, a, M, σM, d, e, f, g):
        oup = e**(d)*g**(-d/2)*gamma(d/2)
        oup += g**(-f/2)*gamma(f/2)
        oup = 2/oup
        return oup


    def f_G(self, a, M, σM, d, e, f, g):
        oup = self.B(a, M, σM, d, e, f, g)
        oup *= ((σM/e)**(-d)+σM**(-f))
        oup *= np.exp(-g/σM**2)
        return oup
    
    
    def rhom_a(self, a):
        ombh2 = self.cosmology['ombh2']
        omch2 = self.cosmology['omch2']
        H0 = self.cosmology['H0'] #[km s^-1 Mpc-1]
        h = H0/100 

        Ωm = ombh2/h**2 + omch2/h**2

        ΩDE = 1 - Ωm
        wDE = self.cosmology['w0'] #'wa' is zero for us

        return Ωm*ρcrit0#*(Ωm*a**(-3) + ΩDE*a**(-3*(1+wDE))) * a**3 # h^2 Msol/Mpc^3

    def M_to_R(self, M, a):
        """
        Converts mass of top-hat filter to radius of top-hat filter

        Parameters:
            - M (float): Mass of the top hat filter in units Msolor/h
            - a (float): Redshift 

        Returns:
            - R (float): Corresponding radius of top hat filter Mpc/h
        """

        return (M / (4/3 * math.pi * self.rhom_a(a))) ** (1/3) # h^-1 Mpc  


    def R_to_M(self, R, a):
        """
        Converts radius of top-hat filter to mass of top-hat filter

        Parameters:
            - R (float): Radius of the top hat filter in units Mpc/h
            - a (float): Redshift 

        Returns:
            - M (float): Corresponding mass of top hat filter Msolar/h 
        """
        return R ** 3 * 4/3 * math.pi * self.rhom_a(a)
    
    def dRdM(self, M, a):
        return 1/(6**(2/3)*np.pi**(1/3)*M**(2/3)*self.rhom_a(a)**(1/3))
