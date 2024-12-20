import sys 
quad_limit = 1000

nu_mass_ev = eval(sys.argv[1])
from aemulusnu_massfunction.emulator_training import *
from aemulusnu_massfunction.fisher_utils import *
from aemulusnu_hmf import massfunction as hmf
from scipy.integrate import quad
fiducial_h = 0.6736


# #(Plank 2018 table 2. TT,TE,EE+lowE+lensing  + neutrino mass put in by hand)
fiducial_cosmology = {'10^9 As':2.1,
                      'ns': 0.9649,
                      'H0': 67.36,
                      'w0': -1,
                      'ombh2': 0.02237,
                      'omch2': 0.12,
                      'nu_mass_ev': nu_mass_ev,}



#(Same as above but put in DES Y3 OmegaM and Sigma8)
print('DES Y3')
Ωmh2 =  0.339*fiducial_h**2 # Y3 3x2pt
Ωνh2 = nu_mass_ev/(93.14) #see astro-ph/0603494
#From the BBN seciton of DES Y3 paper
Ωbh2 = 2.195/100
Ωch2 = Ωmh2-Ωbh2-Ωνh2
fiducial_cosmology = {'10^9 As': 1.520813,  #from σ8 for DES Y3 3x2 and convert_sigma8_to_As.ipynb
                      'ns': 0.9649,
                      'H0': 67.36,
                      'w0': -1,
                      'ombh2': Ωbh2,
                      'omch2': Ωch2,
                      'nu_mass_ev': nu_mass_ev,}

fiducial_cosmo_vals = get_cosmo_vals(fiducial_cosmology)
fiducial_ccl_cosmo = get_ccl_cosmology(tuple(fiducial_cosmo_vals))
fiducial_hmf_cosmology = hmf.cosmology(fiducial_cosmology)







z_bin_edges = [0.2, 0.4, 0.6, 0.8, 1.0]
richness_bin_edges = [20., 30., 45., 60., 300.]



print(fiducial_cosmology)


oup_cov_fname = 'fiducial_cluster_abundance_covariance_'

for key in fiducial_cosmology:
    ckey = key
    if key == '10^9 As':
        ckey = '1e9As'
    oup_cov_fname += '%s_%f_'%(ckey, fiducial_cosmology[key])

oup_cov_fname = list(oup_cov_fname)

for i,char in enumerate(oup_cov_fname):
    if(char == '.'):
        oup_cov_fname[i] = 'p'

oup_cov_fname = oup_cov_fname[:-1]

oup_cov_fname = ''.join(oup_cov_fname)
oup_cov_fname += '.pkl'
print('outputting cov to', oup_cov_fname)



N_fiducial = N_in_z_bins_and_richness_bins(fiducial_cosmology, richness_bin_edges, z_bin_edges)


import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 11






cluster_count_cov = np.zeros((len(z_bin_edges) - 1, len(z_bin_edges) - 1, len(richness_bin_edges) - 1, len(richness_bin_edges) - 1))


halo_bias = ccl.halos.HaloBiasTinker10()


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
    chi, _ = quad(compute_chi_integrand, 0, z_val, epsabs=0, epsrel=1e-3)#units of h^-1 Mpc
    return chi


z_values = np.linspace(0, 2, 500)  # Create an array of z values (adjust range and number of points as needed)
chi_values = [compute_chi(z) for z in z_values]
chi_spline = InterpolatedUnivariateSpline(z_values, chi_values)



from scipy.integrate import quad, dblquad, nquad
from scipy.special import jv 


from functools import cache


θs = np.sqrt(Ωs_rad / np.pi)

@cache
def variance_integral(kperp1, kperp2, z_val):
    kperp = np.sqrt(kperp1**2 + kperp2**2) #units of h / Mpc
    #kperp*h has units 1 / Mpc
    Plin = pkclass.pk_lin(kperp*h, np.array([z_val]))*h**3 #units of Mpc^3/h^3 
    chi = chi_spline(z_val)
    arg = 2*jv(1, kperp*chi*θs) / (kperp*chi*θs) #unitless
    return Plin * arg**2 / (2*np.pi)**2 #units of Mpc^3/h^3 ~ distance^3


@cache
def inner_integral(lam, M, z_val):
    p = cluster_richness_relation(M, lam, z_val)
    dn_dM = emulator(fiducial_hmf_cosmology, M, redshiftToScale(z_val))  # h^4 / Mpc^3 Msun

    bh = halo_bias(fiducial_ccl_cosmo, M / fiducial_h, 1./(1+z_val))
    return p * dn_dM  * bh

    

MAX_K = 10
options={'limit':quad_limit, 'epsrel': 1e-3, 'epsabs': 0}

#############MODS

f_variance_fname = 'f_variance_integrals_'

for key in fiducial_cosmology:
    ckey = key
    if key == '10^9 As':
        ckey = '1e9As'
    f_variance_fname += '%s_%f_'%(ckey, fiducial_cosmology[key])

f_variance_fname = list(f_variance_fname)

for i,char in enumerate(f_variance_fname):
    if(char == '.'):
        f_variance_fname[i] = 'p'

f_variance_fname = f_variance_fname[:-1]

f_variance_fname = ''.join(f_variance_fname)
f_variance_fname += '.pkl'
print('loading variance integral from', f_variance_fname)


f_variance = -1
with open(f_variance_fname, 'rb') as f:
    f_variance = pickle.load(f)


f_richnesses_fname = 'f_richness_integrals_'

for key in fiducial_cosmology:
    ckey = key
    if key == '10^9 As':
        ckey = '1e9As'
    f_richnesses_fname += '%s_%f_'%(ckey, fiducial_cosmology[key])

f_richnesses_fname = list(f_richnesses_fname)

for i,char in enumerate(f_richnesses_fname):
    if(char == '.'):
        f_richnesses_fname[i] = 'p'

f_richnesses_fname = f_richnesses_fname[:-1]

f_richnesses_fname = ''.join(f_richnesses_fname)
f_richnesses_fname += '.pkl'
print('loading richness integral from', f_richnesses_fname)


f_richnesses = -1
with open(f_richnesses_fname, 'rb') as f:
    f_richnesses = pickle.load(f)





def outer_integral(z_val, lam_alpha_min, lam_alpha_max, lam_beta_min, lam_beta_max):
    integral_M_val = f_richnesses[(lam_alpha_min, lam_alpha_max)](z_val)

    integral_M_prime_val = f_richnesses[(lam_beta_min, lam_beta_max)](z_val)


    d2V_dzdOmega = comoving_volume_elements(z_val, cosmo=fiducial_ccl_cosmo)
    h = fiducial_cosmology['H0']/100
    Ωb =  fiducial_cosmology['ombh2'] / h**2
    Ωc =  fiducial_cosmology['omch2'] / h**2

    Ez = np.sqrt((Ωb+Ωc)*(1+z_val)**3 + (1-(Ωb+Ωc))) # unitless
    
    variance = f_variance(z_val)


    return Ωs_rad**2 * integral_M_val * integral_M_prime_val * d2V_dzdOmega**2 * Ez/DH * variance

##################





all_bin_combos = [[i,j,a,b] for i in range(len(z_bin_edges) - 1) 
 for j in range(len(z_bin_edges) - 1) 
 for a in range(len(richness_bin_edges) - 1)
for b in range(len(richness_bin_edges) - 1)]


for i,j,a,b in tqdm(all_bin_combos):
    zi_min = z_bin_edges[i]
    zi_max = z_bin_edges[i + 1]
    zj_min = z_bin_edges[j]
    zj_max = z_bin_edges[j + 1]
    #from Eq(6) of Krause+17, it seems like supersample variance only
    #when the redshift bins overlap. so we can ignore when
    #zi != zj
    if(i != j):
        continue
    la_min = richness_bin_edges[a]
    la_max = richness_bin_edges[a + 1]
    lb_min = richness_bin_edges[b]
    lb_max = richness_bin_edges[b + 1]
    result, error = quad(outer_integral, 
                         zi_min, zi_max, 
                         args=(la_min, la_max, 
                               lb_min, lb_max),
                         limit=quad_limit,
                        epsrel=1e-3, epsabs=0)
#    assert(np.abs(error/result) < 1e-3)
    cluster_count_cov[i,j,a,b] = result 
    if(i == j and a == b): #shot noise
        cluster_count_cov[i,j,a,b] +=  N_fiducial[i][a]


import pickle

with open(oup_cov_fname, 'wb') as file:
        pickle.dump(cluster_count_cov, file)

print('covariance computed')




n_z = len(z_bin_edges) - 1
n_r = len(richness_bin_edges) - 1
cov_matrix_2d = np.zeros((n_z * n_r, n_z * n_r))

for i in range(n_z):
    for j in range(n_z):
        for a in range(n_r):
            for b in range(n_r):
                row_index = i * n_r + a
                col_index = j * n_r + b
                cov_matrix_2d[row_index, col_index] = cluster_count_cov[i, j, a, b]


import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'serif'

plt.figure( dpi=600)
plt.imshow(np.log10(cov_matrix_2d), cmap='rainbow', aspect=1, vmin=-1, vmax=4.5)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.tick_params(
    bottom=False,
    top=True,
labelbottom=False,
labeltop=True)
plt.gca().xaxis.set_ticks_position('none') 
plt.gca().yaxis.set_ticks_position('none') 


# Add ticks for the block divisions
tick_positions_z = np.arange(n_r, n_r * n_z+1, n_r) - 2.5

tick_z_labels = [r'$z^i:[%.1f, %.1f]$'%(a,b) for a,b in zip(z_bin_edges, z_bin_edges[1:])]
plt.xticks(tick_positions_z, tick_z_labels, ha='center')

tick_z_labels = [r'$z^j:[%.1f, %.1f]$'%(a,b) for a,b in zip(z_bin_edges, z_bin_edges[1:])]
plt.yticks(tick_positions_z, tick_z_labels, rotation=90, va='center')

lambda_labels = [r'$\lambda_\alpha:[%d, %d]$'%(a,b) for a,b in zip(richness_bin_edges, richness_bin_edges[1:])]
lambda_labels_0 = [r'$\lambda_\beta:[%d, %d]$'%(a,b) for a,b in zip(richness_bin_edges, richness_bin_edges[1:])]

lw=0.5
for i,l in enumerate(lambda_labels):
    plt.text(3.65 , i,
            s=l,
            horizontalalignment='left',
            verticalalignment='center',
           fontsize=8)
    plt.text(i , 3.65,
            s=lambda_labels_0[i],
            horizontalalignment='center',
            verticalalignment='top',
             rotation=90,
           fontsize=8)
    plt.axvline(i+0.5, .5, .75, linewidth=lw, linestyle='-', color='k')
    plt.axhline(i+0.5, 0.25, 0.5, linewidth=lw, linestyle='-', color='k')

    for j in range(n_z):
        plt.axvline(i+0.5+n_z*j, 0.75-0.25*j,1-0.25*j, linewidth=lw, linestyle='--', color='k')
        plt.axhline(i+0.5+n_z*j, 0+0.25*j,0.25+0.25*j, linewidth=lw, linestyle='--', color='k')

for a in tick_positions_z:
    plt.axvline(a+2, color='black', linestyle='-', linewidth=lw)
    plt.axhline(a+2, color='black', linestyle='-', linewidth=lw)

plt.title(r'$\log_{10}{\rm Cov}(N^i_{\lambda_\alpha} , N^j_{\lambda_\beta})$' + '\n')
# plt.savefig('fiducial_cluster_abundance_cov_nu_mass_%.4f.pdf'%(nu_mass_ev), dpi=600, bbox_inches = "tight")
