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
                      'nu_mass_ev': 0.06,}

fiducial_log10_rel_step_size = { #for numerical derivativese
    '10^9 As': -2.6,
    'ns': -4,
    'H0': -2.3,
    'w0': -2.3,
    'ombh2': -2.6,
    'omch2': -2.3,
    'nu_mass_ev': -1.1,
}

fiducial_cosmo_vals = get_cosmo_vals(fiducial_cosmology)

fiducial_ccl_cosmo = None


#from krause
Ωs = 18000 # deg^2
Ωs_rad = 5/9 * np.pi**2 # rad^2


#for evaluating dM integral
M_min = 1e13
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
tinker08_hmf = ccl.halos.MassFuncTinker08(mass_def='200m')


mass_functions = {'emu': emulator,
                  'st': st_hmf,
                  'b16': bocquet16_hmf,
                  't08': tinker08_hmf}

def cluster_richness_relation(M, λ, z):
    #equation (10) to To, Krause+20
    lnλMean = lnλ0 + Aλ* np.log(M/Mpiv) + Bλ*np.log((1+z)/1.45)

    σintrinsic = 0.3
    σlnλ2 = (σintrinsic**2 + (np.exp(lnλMean) - 1)/np.exp(2*lnλMean))
    σlnλ  = np.sqrt(σlnλ2)
    if(σlnλ2 < 0):
        print(M, σlnλ2, z)

    norm = 1/(np.sqrt(2*np.pi) * λ * σlnλ)
    arg = (np.log(λ) - lnλMean)**2 
    arg /= 2 * σlnλ ** 2
    return norm * np.exp(-arg)

fiducial_ccl_cosmo = get_ccl_cosmology(tuple(fiducial_cosmo_vals))


def comoving_volume_elements(z, cosmo=fiducial_ccl_cosmo):

    h = cosmo['h']
    Ωb =  cosmo['Omega_b']
    Ωc =  cosmo['Omega_c']

    DA = ccl.angular_diameter_distance(cosmo, 1/(1+z)) * h # Mpc / h 
    #According to ccl_background.h, this uses dΩ [radians]
    Ez = np.sqrt((Ωb+Ωc)*(1+z)**3 + (1-(Ωb+Ωc))) # unitless
    return DH*(1+z)**2*DA**2/Ez # h^{-3} Mpc^3 



def cluster_count_integrand(lam, M, z_val, cosmo=fiducial_ccl_cosmo, mf = emulator):
    p = cluster_richness_relation(M, lam, z_val) # h / Msun

    h = cosmo['h']

    dn_dM = mf(cosmo, M/h, redshiftToScale(z_val)) /(h**3 * M * np.log(10)) # h^4 / Mpc^3 Msun
    d2V_dzdOmega = comoving_volume_elements(z_val, cosmo=cosmo) # Mpc^3 / h^3


    return p * dn_dM * d2V_dzdOmega # h / (Msun)


from scipy.integrate import tplquad

def N_in_z_and_richness_bin(lambda_min, lambda_max, z_min, z_max, mf = emulator, cosmo=fiducial_ccl_cosmo):
    cluster_count_integrand_cosmology = partial(cluster_count_integrand, cosmo=cosmo, mf = mf)

    result, error = tplquad(cluster_count_integrand_cosmology, z_min, z_max, M_min, M_max, lambda_min, lambda_max, epsrel=1e-4, epsabs=0)

    if(error/result > .001):
        print(error, result)
    assert(error / result < .001) #.1% accurate

    return Ωs_rad * result

def N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges, mf = emulator):

    N_values = np.zeros((len(z_bin_edges) - 1, len(richness_bin_edges) - 1))

    cosmo_vals = tuple(get_cosmo_vals(cosmology))
    cosmo = get_ccl_cosmology(cosmo_vals)

    print(mf.name)

    for i in range(len(z_bin_edges) - 1):
        print('redshift bin %d of %d'%(i+1, len(z_bin_edges)-1))
        z_min = z_bin_edges[i]
        z_max = z_bin_edges[i + 1]

        for j in trange(len(richness_bin_edges) - 1):
            lambda_min = richness_bin_edges[j]
            lambda_max = richness_bin_edges[j + 1]

            # Evaluate the function for the given bin
            N_values[i, j] = N_in_z_and_richness_bin(lambda_min, lambda_max, z_min, z_max, cosmo=cosmo, mf = mf)

    return N_values
