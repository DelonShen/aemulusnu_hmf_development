from aemulusnu_massfunction.emulator_training import *
from aemulusnu_massfunction.fisher_utils import *
import pickle
import sys

parameter_changed = sys.argv[1]
parameter_log10_rel_step_size = eval(sys.argv[2])

fiducial_h = 0.6736


# #(Plank 2018 table 2. TT,TE,EE+lowE+lensing  + neutrino mass put in by hand)
fiducial_cosmology = {'10^9 As':2.1,
                      'ns': 0.9649,
                      'H0': 67.36,
                      'w0': -1,
                      'ombh2': 0.02237,
                      'omch2': 0.12,
                      'nu_mass_ev': 0.06,}



#(Same as above but put in DES Y3 OmegaM and Sigma8)
print('DES Y3')
Ωmh2 =  0.339*fiducial_h**2 # Y3 3x2pt
Ωνh2 = 0.06/(93.14) #see astro-ph/0603494
#From the BBN seciton of DES Y3 paper
Ωbh2 = 2.195/100
Ωch2 = Ωmh2-Ωbh2-Ωνh2
fiducial_cosmology = {'10^9 As': 1.520813,  #from σ8 for DES Y3 3x2 and convert_sigma8_to_As.ipynb
                      'ns': 0.9649,
                      'H0': 67.36,
                      'w0': -1,
                      'ombh2': Ωbh2,
                      'omch2': Ωch2,
                      'nu_mass_ev': 0.06,}






fiducial_cosmo_vals = get_cosmo_vals(fiducial_cosmology)
fiducial_ccl_cosmo = get_ccl_cosmology(tuple(fiducial_cosmo_vals))



if(len(sys.argv) == 4):
    nu_mass_ev = eval(sys.argv[3])
    print('changing fiducial neutrino mass to', nu_mass_ev)

    fiducial_cosmology = {'10^9 As':2.1,
                          'ns': 0.9649,
                          'H0': 67.36,
                          'w0': -1,
                          'ombh2': 0.02237,
                          'omch2': 0.12,
                          'nu_mass_ev': nu_mass_ev,}

    fiducial_cosmo_vals = get_cosmo_vals(fiducial_cosmology)
    fiducial_ccl_cosmo = get_ccl_cosmology(tuple(fiducial_cosmo_vals))


print(parameter_changed, parameter_log10_rel_step_size)



oup_fname = '/scratch/users/delon/aemulusnu_massfunction/cluster_abundance_tinker_fisher_changing_%s_log10rel_step%.4f_cosmo_'%(parameter_changed, parameter_log10_rel_step_size)

for key in fiducial_cosmology:
    ckey = key
    if key == '10^9 As':
        ckey = '1e9As'
    oup_fname += '%s_%f_'%(ckey, fiducial_cosmology[key])

oup_fname = list(oup_fname)

for i,char in enumerate(oup_fname):
    if(char == '.'):
        oup_fname[i] = 'p'

oup_fname = oup_fname[:-1]

oup_fname = ''.join(oup_fname)

print(oup_fname)





z_bin_edges = [0.2, 0.4, 0.6, 0.8, 1.0]
richness_bin_edges = [20., 30., 45., 60., 300.]

print('fiducial: ')
print(fiducial_cosmology)

cosmology = fiducial_cosmology.copy()
cosmology[parameter_changed] += np.abs(cosmology[parameter_changed]) * 10**parameter_log10_rel_step_size

print('pos step')
print(cosmology)

print('using tinker08')
cluster_abundance_pos = N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges, mf=tinker08_hmf)



with open(oup_fname, 'wb') as file:
    print(oup_fname)
    pickle.dump(cluster_abundance_pos, file)



cosmology = fiducial_cosmology.copy()
cosmology[parameter_changed] -= np.abs(cosmology[parameter_changed]) * 10**parameter_log10_rel_step_size

print('neg step')
print(cosmology)


print('using tinker08')
cluster_abundance_neg = N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges,  mf=tinker08_hmf)


oup_fname += '_neg'

with open(oup_fname, 'wb') as file:
    print(oup_fname)
    pickle.dump(cluster_abundance_neg, file)
