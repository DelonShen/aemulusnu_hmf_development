import math
from scipy.integrate import quad, fixed_quad
import matplotlib.pyplot as plt
import pickle
import numpy as np
import functools

ρcrit0 = 2.77533742639e+11 #h^2 Msol / Mpc^3
cosmo_params = pickle.load(open('data/cosmo_params.pkl', 'rb'))

def scaleToRedshift(a):
    return 1/a-1

def redshiftToScale(z):
    return 1/(1+z)



def dσ2dk(k, R, pk):
    x = k * R
    W = (3 / x) * (np.sin(x) / x**2 - np.cos(x) / x)
    return W**2 * pk(k) * k**2 / 2 / np.pi**2

def dσ2dRdk(k, R, pk):
    x = k * R
    W = (3 / x) * (np.sin(x) / x**2 - np.cos(x) / x)
    dWdx = (-3 / x) * ((3 / x**2 - 1) * np.sin(x) / x - 3 * np.cos(x) / x**2)
    return 2 * W * dWdx * pk(k) * k**3 / 2 / np.pi**2

@functools.cache
def sigma2_scalar(pk, R):
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
    res, err = quad(lambda k: dσ2dk(k, R, pk), 0, 20/R, limit=1000)
    return res

@functools.cache
def dsigma2dR_scalar(pk, R):
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

    res, err = quad(lambda k: dσ2dRdk(k, R, pk), 0, 20/R, limit=1000)
    return res

sigma2 = np.vectorize(sigma2_scalar)
dsigma2dR = np.vectorize(dsigma2dR_scalar)

class Normalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, X):
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)

    def transform(self, X):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer has not been fitted. Call fit() first.")

        return (X - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, X_normalized):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer has not been fitted. Call fit() first.")
        return X_normalized * (self.max_val - self.min_val) + self.min_val


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer has not been fitted. Call fit() first.")
        return (X - self.mean) / self.std

    def inverse_transform(self, X_std):
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer has not been fitted. Call fit() first.")
        return X_std * self.std + self.mean
