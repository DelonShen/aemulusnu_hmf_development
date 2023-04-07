import math
from scipy.integrate import quad
import matplotlib.pyplot as plt

rhocrit = 2.77533742639e+11  # critical density of the universe in units of h^2 M_sun / Mpc^3

def M_to_R(M, Omega_m):
    return (M / (4/3 * math.pi * rhocrit * Omega_m)) ** (1/3)

def R_to_M(R, Omega_m):
    return R ** 3 * 4/3 * math.pi * rhocrit * Omega_m

def scaleToRedshift(a):
    return 1/a-1

def sigma2(pk, R):
    def dσ2dk(k):
        x = k * R
        W = (3 / x) * (np.sin(x) / x**2 - np.cos(x) / x)
        dσ2dk = W**2 * pk(k) * k**2 / 2 / np.pi**2
        return dσ2dk
    res, err = quad(dσ2dk, 0, 20 / R)
    σ2 = res
    return σ2