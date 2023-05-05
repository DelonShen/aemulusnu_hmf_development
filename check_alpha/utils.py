import math
from scipy.integrate import quad, fixed_quad
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import functools

ρcrit0 = 2.77533742639e+11 #h^2 Msol / Mpc^3
cosmo_params = pickle.load(open('data/cosmo_params.pkl', 'rb'))

def M_to_R(M, box, a):
    """
    Converts mass of top-hat filter to radius of top-hat filter
    
    Parameters:
        - M (float): Mass of the top hat filter in units Msolor/h
        - box (string): Which Aemulus nu box we're considering 
        - a (float): Redshift 

    Returns:
        - R (float): Corresponding radius of top hat filter Mpc/h
    """

    return (M / (4/3 * math.pi * rhom_a(box, a))) ** (1/3) # h^-1 Mpc  

def R_to_M(R,box, a):
    """
    Converts radius of top-hat filter to mass of top-hat filter
    
    Parameters:
        - R (float): Radius of the top hat filter in units Mpc/h
        - box (string): Which Aemulus nu box we're considering 
        - a (float): Redshift 

    Returns:
        - M (float): Corresponding mass of top hat filter Msolar/h 
    """
    return R ** 3 * 4/3 * math.pi * rhom_a(box, a)

def scaleToRedshift(a):
    return 1/a-1

def redshiftToScale(z):
    return 1/(1+z)

@functools.cache
def sigma2(pk, R):
    """
    Adapated from https://github.com/komatsu5147/MatterPower.jl
    Computes variance of mass fluctuations with top hat filter of radius R
    For this function let k be the comoving wave number with units h/Mpc

    Parameters:
        - pk (funtion): P(k), the matter power spectrum which has units Mpc^3 / h^3
        - R (float): The smoothing scale in units Mpc/h
    Returns:
        - sigma2 (float): The variance of mass fluctuations
    """

    def dσ2dk(k):
        x = k * R
        W = (3 / x) * (np.sin(x) / x**2 - np.cos(x) / x)
        dσ2dk = W**2 * pk(k) * k**2 / 2 / np.pi**2
        return dσ2dk
    res, err = quad(dσ2dk, 0, np.inf)
    σ2 = res
    return σ2

def rhom_a(box, a):
    ombh2 = cosmo_params[box]['ombh2']
    omch2 = cosmo_params[box]['omch2']
    H0 = cosmo_params[box]['H0'] #[km s^-1 Mpc-1]
    h = H0/100 

    Ωm = ombh2/h**2 + omch2/h**2
    
    ΩDE = 1 - Ωm
    wDE = cosmo_params[box]['w0'] #'wa' is zero for us

    return Ωm*ρcrit0#*(Ωm*a**(-3) + ΩDE*a**(-3*(1+wDE))) * a**3 # h^2 Msol/Mpc^3
    
@functools.cache
def dsigma2dR(pk, R):
    """
    Adapated from https://github.com/komatsu5147/MatterPower.jl
    Computes deriative of variance of mass fluctuations wrt top hat filter of radius R
    For this function let k be the comoving wave number with units h/Mpc
    
    Parameters:
        - pk (funtion): P(k), the matter power spectrum which has units Mpc^3 / h^3
        - R (float): The smoothing scale in units Mpc/h
    Returns:
        - dsigma2dR (float): The derivative of the variance of mass fluctuations wrt R
    """

    def dσ2dRdk(k):
        x = k * R
        W = (3 / x) * (np.sin(x) / x**2 - np.cos(x) / x)
        dWdx = (-3 / x) * ((3 / x**2 - 1) * np.sin(x) / x - 3 * np.cos(x) / x**2)
        dσ2dRdk = 2 * W * dWdx * pk(k) * k**3 / 2 / np.pi**2
        return dσ2dRdk
    res, err = quad(dσ2dRdk, 0, np.inf)
    return res

def dRdM(M, box, a):
    return 1/(6**(2/3)*np.pi**(1/3)*M**(2/3)*rhom_a(box, a)**(1/3))