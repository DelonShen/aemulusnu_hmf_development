from .emulator import *
import pyccl as ccl

from scipy.interpolate import RectBivariateSpline

from functools import cache, partial

emulator = AemulusNu_HMF_Emulator()

# c / H 
DH = 3000 #[h^{-1} Mpc] 

#fiducial cosmolgy
#(Plank 2018 (base LCDM) + neutrino mass put in by hand)
#Table 1.
fiducial_cosmology = {'10^9 As':2.09681,
                      'ns': 0.9652,
                      'H0': 67.37,
                      'w0': -1,
                      'ombh2': 0.02233,
                      'omch2': 0.1198,
                      'nu_mass_ev': 0.07,}

fiducial_log10_rel_step_size = { #for numerical derivativese
    '10^9 As': -2.6,
    'ns': -4,
    'H0': -2.3,
    'w0': -2.3,
    'ombh2': -2.2,
    'omch2': -2.6,
    'nu_mass_ev': -1.08,
}

fiducial_cosmo_vals = emulator.get_cosmo_vals(fiducial_cosmology)

fiducial_ccl_cosmo = None


#from krause
Ωs = 18000 # deg^2
Ωs_rad = 5/9 * np.pi**2 # rad^2


#for evaluating dM integral
M_min = 1e11
M_max = 1e16


#for cluster richness relation
#reading valaues from figure B1 of To, Krause+20, 
#TODO get actual ML values
Aλ = .9
lnλ0 = 3.8
Bλ = -0.4
Mpiv = 5e14 # h^-1 M_sol


#misc
st_hmf = ccl.halos.MassFuncSheth99(mass_def='200m', mass_def_strict=False)
bocquet16_hmf= ccl.halos.MassFuncBocquet16(mass_def='200m')
def cluster_richness_relation(M, λ, z):
    #equation (10) to To, Krause+20
    lnλMean = lnλ0 + Aλ* np.log(M/Mpiv) + Bλ*np.log((1+z)/1.45)

    σintrinsic = 0.3
    σlnλ = np.sqrt(σintrinsic) #im simplifying eq(9) of To, Krause+20, it seems like second term is small? 

    norm = 1/(np.sqrt(2*np.pi) * λ * σlnλ)
    arg = (np.log(λ) - lnλMean)**2 
    arg /= 2 * σlnλ ** 2
    return norm * np.exp(-arg)

@cache
def get_ccl_cosmology(cosmo_vals):
    cosmology = emulator.get_cosmo_dict(cosmo_vals)

    h = cosmology['H0']/100
    Ωb =  cosmology['ombh2'] / h**2
    Ωc =  cosmology['omch2'] / h**2

    cosmo = ccl.Cosmology(Omega_c=Ωc,
                          Omega_b=Ωb,
                          h=h,
                          A_s=cosmology['10^9 As']*10**(-9),
                          n_s=cosmology['ns'])

    return cosmo

fiducial_ccl_cosmo = get_ccl_cosmology(tuple(fiducial_cosmo_vals))


def comoving_volume_elements(z, cosmo_vals):
    cosmology = emulator.get_cosmo_dict(cosmo_vals)

    h = cosmology['H0']/100
    Ωb =  cosmology['ombh2'] / h**2
    Ωc =  cosmology['omch2'] / h**2

    cosmo = get_ccl_cosmology(cosmo_vals)

    DA = ccl.angular_diameter_distance(cosmo, 1/(1+z)) * h # Mpc / h 
    #According to ccl_background.h, this uses dΩ [radians]
    Ez = np.sqrt((Ωb+Ωc)*(1+z)**3 + (1-(Ωb+Ωc))) # unitless
    return DH*(1+z)**2*DA**2/Ez # h^{-3} Mpc^3 



def cluster_count_integrand(lam, M, z_val, cosmo_vals, sheth_tormen=False, bocquet16=False):
    p = cluster_richness_relation(M, lam, z_val) # h / Msun

    dn_dM = emulator.predict_dndM(emulator.get_cosmo_dict(cosmo_vals), z_val, M) # h^4 / Mpc^3 Msun
    if(sheth_tormen):
        cosmo = get_ccl_cosmology(cosmo_vals)
        cosmology = emulator.get_cosmo_dict(cosmo_vals)
        h = cosmology['H0']/100
        dn_dM = st_hmf(cosmo, M/h, redshiftToScale(z_val)) / (h**4 * M * np.log(10))
    elif(bocquet16):
        cosmo = get_ccl_cosmology(cosmo_vals)
        cosmology = emulator.get_cosmo_dict(cosmo_vals)
        h = cosmology['H0']/100
        dn_dM = bocquet16_hmf(cosmo, M/h, redshiftToScale(z_val)) / (h**4 * M * np.log(10))

    d2V_dzdOmega = comoving_volume_elements(z_val, cosmo_vals) # Mpc^3 / h^3


    return p * dn_dM * d2V_dzdOmega # h / (Msun)


from scipy.integrate import tplquad

def N_in_z_and_richness_bin(cosmology, lambda_min, lambda_max, z_min, z_max,
                            sheth_tormen=False, bocquet16=False):
    cluster_count_integrand_cosmology = partial(cluster_count_integrand, cosmo_vals = tuple(emulator.get_cosmo_vals(cosmology)), 
                                                sheth_tormen=sheth_tormen, bocquet16=bocquet16)

    result, error = tplquad(cluster_count_integrand_cosmology, z_min, z_max, M_min, M_max, lambda_min, lambda_max, epsrel=1e-4, epsabs=0)

    assert(error / result < .001) #.1% accurate

    return Ωs_rad * result

def N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges, sheth_tormen=False, bocquet16=False):

    N_values = np.zeros((len(z_bin_edges) - 1, len(richness_bin_edges) - 1))

    for i in trange(len(z_bin_edges) - 1):
        z_min = z_bin_edges[i]
        z_max = z_bin_edges[i + 1]

        for j in range(len(richness_bin_edges) - 1):
            lambda_min = richness_bin_edges[j]
            lambda_max = richness_bin_edges[j + 1]

            # Evaluate the function for the given bin
            N_values[i, j] = N_in_z_and_richness_bin(cosmology, lambda_min, lambda_max, z_min, z_max, sheth_tormen=sheth_tormen,  bocquet16=bocquet16)

    return N_values