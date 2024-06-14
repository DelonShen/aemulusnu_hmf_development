from .emulator_training import *
import pyccl as ccl

from scipy.interpolate import RectBivariateSpline

from functools import cache, partial

emulator = AemulusNu_HMF_Emulator()



from pyccl.halos import MassFunc, get_delta_c

class MassFuncTinker08Costanzi13(MassFunc):
    """Implements the mass function of `Tinker et al. 2008
    <https://arxiv.org/abs/0803.2706>`_. This parametrization accepts S.O.
    masses with :math:`200 < \\Delta < 3200`, defined with respect to the
    matter density. This can be automatically translated to S.O. masses
    defined with respect to the critical density.

    Modified to use Costanzi et al. 2013 (JCAP12(2013)012) nuCDM HMF prescription

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
    """
    name = 'custom Tinker08'

    def __init__(self, *,
                 mass_def="200m",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
    def _get_logM_sigM(self, cosmo, M, a, *, return_dlns=False):
        """Compute ``logM``, ``sigM``, and (optionally) ``dlns_dlogM``.
            Using Costanzi et al. 2013 (JCAP12(2013)012) perscription
            to evaluate HMF in nuCDM cosmology, we replace P_m with P_cb
            """
        if('mirror_cosmo' not in cosmo['extra_parameters']):
            self.init_cosmo(cosmo)

        mirror_cosmo = cosmo['extra_parameters']['mirror_cosmo']
        mirror_cosmo.compute_sigma()  # initialize sigma(M) splines if needed
        logM = np.log10(M)
        # sigma(M)
        status = 0
        sigM, status = lib.sigM_vec(mirror_cosmo.cosmo, a, logM, len(logM), status)
        check(status, cosmo=mirror_cosmo)
        if not return_dlns:
            return logM, sigM

        # dlogsigma(M)/dlog10(M)
        dlns_dlogM, status = lib.dlnsigM_dlogM_vec(mirror_cosmo.cosmo, a, logM,
                                                   len(logM), status)
        check(status, cosmo=mirror_cosmo)
        return logM, sigM, dlns_dlogM

    def init_cosmo(self, cosmo):
        cosmo['extra_parameters']['mirror_cosmo'] = ccl.Cosmology(Omega_c=cosmo['Omega_c'],
                                                 Omega_b=cosmo['Omega_b'],
                                                 h=cosmo['h'],
                                                 A_s=cosmo['A_s'],
                                                 n_s=cosmo['n_s'],
                                                 w0=cosmo['w0'],
                                                 m_nu=cosmo['m_nu'])
        funcType = type(cosmo['extra_parameters']['mirror_cosmo']._compute_linear_power)

        cosmo['extra_parameters']['mirror_cosmo']._compute_linear_power = MethodType(custom_compute_linear_power,
                                                                                     cosmo['extra_parameters']['mirror_cosmo'])
        
    def __call__(self, cosmo, M, a):
        """ Returns the mass function for input parameters. 
            Using Costanzi et al. 2013 (JCAP12(2013)012) perscription
            to evaluate HMF in nuCDM cosmology

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
            M (:obj:`float` or `array`): halo mass.
            a (:obj:`float`): scale factor.

        Returns:
            (:obj:`float` or `array`): mass function \
                :math:`dn/d\\log_{10}M` in units of Mpc^-3 (comoving).
        """
        if('mirror_cosmo' not in cosmo['extra_parameters']):
            self.init_cosmo(cosmo)
            
        M_use = np.atleast_1d(M)
        logM, sigM, dlns_dlogM = self._get_logM_sigM( 
            cosmo, M_use, a, return_dlns=True)

        rho = (const.RHO_CRITICAL
               * (cosmo['Omega_c'] + cosmo['Omega_b'])
               * cosmo['h']**2)

        f = self._get_fsigma(cosmo, sigM, a, 2.302585092994046 * logM)
        mf = f * rho * dlns_dlogM / M_use
        if np.ndim(M) == 0:
            return mf[0]
        return mf


    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "200m"

    def _setup(self):
        delta = np.array(
            [200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.])
        alpha = np.array(
            [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
        beta = np.array(
            [1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        gamma = np.array(
            [2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        phi = np.array(
            [1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
        ldelta = np.log10(delta)
        self.pA0 = interp1d(ldelta, alpha)
        self.pa0 = interp1d(ldelta, beta)
        self.pb0 = interp1d(ldelta, gamma)
        self.pc = interp1d(ldelta, phi)

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        ld = np.log10(self.mass_def._get_Delta_m(cosmo, a))
        pA = self.pA0(ld) * a**0.14
        pa = self.pa0(ld) * a**0.06
        pd = 10.**(-(0.75/(ld - 1.8750612633))**1.2)
        pb = self.pb0(ld) * a**pd
        return pA * ((pb / sigM)**pa + 1) * np.exp(-self.pc(ld)/sigM**2)
    
    
    
    
# c / H 
DH = 3000 #[h^{-1} Mpc] 

#fiducial cosmolgy
#(Plank 2018 table 2. TT,TE,EE+lowE+lensing  + neutrino mass put in by hand)
#Table 1.
fiducial_cosmology = {'10^9 As':2.1,
                      'ns': 0.9649,
                      'H0': 67.36,
                      'w0': -1,
                      'ombh2': 0.02237,
                      'omch2': 0.12,
                      'nu_mass_ev': 0.06,}

fiducial_log10_rel_step_size = { #for numerical derivativese
    '10^9 As': -2.6,
    'ns': -4,
    'H0': -2.3,
    'w0': -2.3,
    'ombh2': -2.6,
    'omch2': -2.3,
    'nu_mass_ev': -2.3,
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
# st_hmf = ccl.halos.MassFuncSheth99(mass_def='200m', mass_def_strict=False)
# bocquet16_hmf= ccl.halos.MassFuncBocquet16(mass_def='200m')
tinker08_hmf =MassFuncTinker08Costanzi13(mass_def='200m')


mass_functions = {'emu': emulator,
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
