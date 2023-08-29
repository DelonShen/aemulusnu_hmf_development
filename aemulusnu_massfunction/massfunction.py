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


class MassFunction:
    def __init__(self, cosmology, fixed={}):
        '''
        Comoving halo mass function

        '''
        self.cosmology = cosmology
        self.fixed = fixed

#        print('Setting dictionary')
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
            'A_s': cosmo['10^9 As'] * 10**-9,
            'n_s': cosmo['ns'],
            'Omega_Lambda': 0.0,
            'w0_fld': cosmo['w0'],
            'wa_fld': 0.0,
            'cs2_fld': 1.0,
            'fluid_equation_of_state': "CLP"
        }

#        print('Computing sigma spline')
        #get logsigma spline
        M = 10**np.linspace(11, 17, 300)
        z = np.linspace(0, 2, 100)
        # Create meshgrid
        M_grid, z_grid = np.meshgrid(M, z)

#        print('Initializing CLASS')
        pkclass = Class()
        pkclass.set(cosmo_dict)
        pkclass.compute()

        R = self.M_to_R(M, redshiftToScale(z)) #h^-1 Mpc

        h = self.cosmology['H0']/100


        for z_curr in z:
            kt = np.logspace(-3, 1, 100) # h/Mpc
            pk_m_lin = np.array(
                [
                    pkclass.pk_lin(ki, np.array([z_curr]))*h**3 #units of Mpc^3/h^3
                    for ki in kt * h # 1 / Mpc
                ]
            )

        #compute sigma on this mesh
        sigma = np.zeros_like(M_grid)
#        print('Computing Sigma from CLASS')
        for i in range(len(z)):
            for j in range(len(R)):
                sigma[i, j] = pkclass.sigma(R[j], z[i], h_units=True) #h^-1 Mpc


        # Fit the spline
        self.f_logsigma_logM = RectBivariateSpline(z, np.log(M), np.log(sigma))
#        print('Spline Made')




    def dndM(self, a, M, d, e, f, g):
        R = self.M_to_R(M, a) #Mpc/h
        σM = np.exp(self.f_logsigma_logM(scaleToRedshift(a), np.log(M)))[0][0]
        oup = self.f_G(σM, d, e, f, g) #unitless
        oup *= self.rhom_a(a)/M**2 # h^4 /Mpc^3 Msun

        dlogsiginv_dlogM = -self.f_logsigma_logM.ev(scaleToRedshift(a), np.log(M), dy=1)

        oup *= dlogsiginv_dlogM
        return oup # h^4 / (Mpc^3  Msun)


    @cache
    def B(self, d, e, f, g):
        oup = e**(d)*g**(-d/2)*gamma(d/2)
        oup += g**(-f/2)*gamma(f/2)
        oup = 2/oup
        return oup



    @cache
    def f_G(self, σM, d, e, f, g):
        oup = self.B(d, e, f, g)
        oup *= ((σM / e) ** (-d) + σM ** (-f))
        oup *= np.exp(-g / σM ** 2)
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
