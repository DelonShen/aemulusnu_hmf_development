from scipy.special import gamma
from scipy.optimize import curve_fit
from utils import *

d0 = 2.4
def p(a, p0, p1, p2, p3):
    oup = (p0)+(a-0.5)*(p1)+(a-0.5)**2*(p2)+(a-0.5)**3*(p3)
    return oup

def B(a, M, σM, d, e, f, g):
    oup = e**(d)*g**(-d/2)*gamma(d/2)
    oup += g**(-f/2)*gamma(f/2)
    oup = 2/oup
    return oup
    
    
def f_G(a, M, σM, d, e, f, g):
    oup = B(a, M, σM, d, e, f, g)
    oup *= ((σM/e)**(-d)+σM**(-f))
    oup *= np.exp(-g/σM**2)
    return oup

def tinker(a, M, 
            d1, d2, d3,
           e0, e1, e2, e3,
           f0, f1, f2, f3,
           g0, g1, g2, g3,
           dlnσinvdM, Pk, R, rhobm):
    d = p(a, d0, d1, d2, d3)
    e = p(a, e0, e1, e2, e3)
    f = p(a, f0, f1, f2, f3)
    g = p(a, g0, g1, g2, g3)
    
    σM = np.sqrt(sigma2(Pk, R))
    oup = f_G(a, M, σM, d, e, f, g)
    oup *= rhobm/M
    oup *= dlnσinvdM(M)
    return oup