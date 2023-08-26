from aemulusnu_massfunction.emulator import *
from aemulusnu_massfunction.fisher_utils import *

z_bin_edges = [0.2, 0.4, 0.6, 0.8, 1.0]
richness_bin_edges = [20., 30., 45., 60., 300.]

N_fiducial = N_in_z_bins_and_richness_bins(fiducial_cosmology, richness_bin_edges, z_bin_edges)


cluster_count_cov = np.zeros((len(z_bin_edges) - 1, len(z_bin_edges) - 1, len(richness_bin_edges) - 1, len(richness_bin_edges) - 1))

halo_bias = ccl.halos.HaloBiasTinker10(fiducial_ccl_cosmo)

fiducial_h = fiducial_cosmology['H0']/100
halo_bias.get_halo_bias(fiducial_ccl_cosmo, 1e14 *  fiducial_h, 1) #[Mass] is Msun / h

from classy import Class

h = fiducial_cosmology['H0']/100
cosmo_dict = {
    'h': h,
    'Omega_b': fiducial_cosmology['ombh2'] / h**2,
    'Omega_cdm': fiducial_cosmology['omch2'] / h**2,
    'N_ur': 0.00641,
    'N_ncdm': 1,
    'output': 'mPk mTk',
    'z_pk': '0.0,99',
    'P_k_max_h/Mpc': 20.,
    'm_ncdm': fiducial_cosmology['nu_mass_ev']/3,
    'deg_ncdm': 3,
    'T_cmb': 2.7255,
    'A_s': fiducial_cosmology['10^9 As'] * 10**-9,
    'n_s': fiducial_cosmology['ns'],
    'Omega_Lambda': 0.0,
    'w0_fld': fiducial_cosmology['w0'],
    'wa_fld': 0.0,
    'cs2_fld': 1.0,
    'fluid_equation_of_state': "CLP"
}

#get logsigma spline
z = np.linspace(0, 2, 100)

pkclass = Class()
pkclass.set(cosmo_dict)
pkclass.compute()

from scipy.interpolate import InterpolatedUnivariateSpline

def compute_chi_integrand(z_val):
    Ωb =  fiducial_cosmology['ombh2'] / h**2
    Ωc =  fiducial_cosmology['omch2'] / h**2
    Ez = np.sqrt((Ωb+Ωc)*(1+z_val)**3 + (1-(Ωb+Ωc))) # unitless
    return DH/Ez #units of distance h^-1 Mpc
def compute_chi(z_val):
    chi, _ = quad(compute_chi_integrand, 0, z_val, epsabs=0, epsrel=1e-4)#units of h^-1 Mpc
    return chi


z_values = np.linspace(0, 2, 500)  # Create an array of z values (adjust range and number of points as needed)
chi_values = [compute_chi(z) for z in z_values]
chi_spline = InterpolatedUnivariateSpline(z_values, chi_values)


from scipy.integrate import quad, dblquad
from scipy.special import spherical_jn as jn

from functools import cache


    
@cache
def variance_integral(kperp1, kperp2, z_val):
    kperp = np.sqrt(kperp1**2 + kperp2**2) #units of h / Mpc
    #kperp*h has units 1 / Mpc
    Plin = pkclass.pk_lin(kperp*h, np.array([z_val]))*h**3 #units of Mpc^3/h^3 
    chi = chi_spline(z_val)
    arg = 2*jn(1, kperp*chi*θs) / (kperp*chi*θs) #unitless
    return Plin * arg**2 / (2*np.pi)**2 #units of Mpc^3/h^3 ~ distance^3

@cache
def inner_integral(lam, M, z_val):
    p = cluster_richness_relation(M, lam, z_val)
    dn_dM = emulator.predict_dndM(fiducial_cosmology, z_val, M)
    bh = halo_bias.get_halo_bias(fiducial_ccl_cosmo, M * fiducial_h, 1./(1+z_val))
    return p * dn_dM  * bh

    

MAX_K = 10
def outer_integral(z_val, lam_alpha_min, lam_alpha_max, lam_beta_min, lam_beta_max):
    integral_M_val, _ = dblquad(inner_integral, M_min, M_max, lam_alpha_min, lam_alpha_max,
                             args=(z_val,), 
                             epsrel=1e-4, epsabs=0)

    integral_M_prime_val, _ = dblquad(inner_integral, M_min, M_max, lam_beta_min, lam_beta_max,
                             args=(z_val,), 
                             epsrel=1e-4, epsabs=0)
    
    d2V_dzdOmega = comoving_volume_elements(z_val, tuple(fiducial_cosmo_vals))
    h = fiducial_cosmology['H0']/100
    Ωb =  fiducial_cosmology['ombh2'] / h**2
    Ωc =  fiducial_cosmology['omch2'] / h**2

    Ez = np.sqrt((Ωb+Ωc)*(1+z_val)**3 + (1-(Ωb+Ωc))) # unitless
    
    variance, _ = dblquad(variance_integral, 0, MAX_K, 0, MAX_K, args=(z_val,), epsrel=1e-4, epsabs=0)

    return Ωs_rad**2 * integral_M_val * integral_M_prime_val * d2V_dzdOmega**2 * Ez/DH * variance








import sys
i, j, a, b = map(eval, sys.argv[1:])
print(i, j, a, b)
oup = 0

zi_min = z_bin_edges[i]
zi_max = z_bin_edges[i + 1]
zj_min = z_bin_edges[j]
zj_max = z_bin_edges[j + 1]
#from Eq(6) of Krause+17, it seems like supersample variance only
#when the redshift bins overlap. so we can ignore when
#zi != zj
print('computing covariance element')
if(i == j):
    la_min = richness_bin_edges[a]
    la_max = richness_bin_edges[a + 1]
    lb_min = richness_bin_edges[b]
    lb_max = richness_bin_edges[b + 1]
    result, error = quad(outer_integral,
                         zi_min, zi_max,
                         args=(la_min, la_max,
                               lb_min, lb_max),
                        epsrel=1e-4, epsabs=0)
    oup = result
    if(i == j and a == b): #shot noise
        oup +=  N_fiducial[i][a]

#write to file
with open('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/cluster_cov_i%d_j%d_a%d_b%d.pkl'%(i, j, a, b), 'wb') as f:
    pickle.dump(oup, f)
