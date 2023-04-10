import math
from scipy.integrate import quad, fixed_quad
import matplotlib.pyplot as plt
import pickle 
import numpy as np

G = 4.3009e-9 #km^2 Mpc/ (Msolar  s^2) weird units to make rhom_a good units 
cosmo_params = pickle.load(open('cosmo_params.pkl', 'rb'))

def M_to_R(M, box, a):
    return (M / (4/3 * math.pi * rhom_a(box, a))) ** (1/3) # h^-1 Mpc  

def R_to_M(R,box, a):
    return R ** 3 * 4/3 * math.pi * rhom_a(box, a)

def scaleToRedshift(a):
    return 1/a-1

def redshiftToScale(z):
    return 1/(1+z)

def sigma2(pk, R):
    def dσ2dk(k):
        x = k * R
        W = (3 / x) * (np.sin(x) / x**2 - np.cos(x) / x)
        dσ2dk = W**2 * pk(k) * k**2 / 2 / np.pi**2
        return dσ2dk
    res, err = fixed_quad(dσ2dk, 0, 20 / R)
    σ2 = res
    return σ2

def rhom_a(box, a):
    ombh2 = cosmo_params[box]['ombh2']
    omch2 = cosmo_params[box]['omch2']
    H0 = cosmo_params[box]['H0'] #[km s^-1 Mpc-1]
    h = H0/100
    
    Ωm = ombh2/h**2 + omch2/h**2
    ΩΛ = 1 - Ωm
    ρcrit0 = 3*H0**2/(8*np.pi*G) # h^2 Msol/Mpc^3
    
    return Ωm*ρcrit0*(Ωm*a**(-3) + ΩΛ) 
    
    