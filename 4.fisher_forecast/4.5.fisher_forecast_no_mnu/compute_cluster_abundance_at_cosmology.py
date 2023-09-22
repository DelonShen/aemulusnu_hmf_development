from aemulusnu_massfunction.emulator import *
from aemulusnu_massfunction.fisher_utils import *
import pickle
import sys




fiducial_cosmology = {'10^9 As':2.09681,
                      'ns': 0.9652,
                      'H0': 67.37,
                      'w0': -1,
                      'ombh2': 0.02233,
                      'omch2': 0.1198,
                      'nu_mass_ev': 0.00,} #no neutrino mass

fiducial_cosmo_vals = get_cosmo_vals(fiducial_cosmology)

fiducial_ccl_cosmo = get_ccl_cosmology(tuple(fiducial_cosmo_vals))
print(fiducial_cosmology)

parameter_changed = sys.argv[1]
parameter_log10_rel_step_size = eval(sys.argv[2])



print(parameter_changed, parameter_log10_rel_step_size)


oup_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/cluster_abundance_fisher_no_mnu_%s_%.4f'%(parameter_changed, parameter_log10_rel_step_size)

print(oup_fname)





z_bin_edges = [0.2, 0.4, 0.6, 0.8, 1.0]
richness_bin_edges = [20., 30., 45., 60., 300.]

print('fiducial: ')
print(fiducial_cosmology)

cosmology = fiducial_cosmology.copy()
cosmology[parameter_changed] += np.abs(cosmology[parameter_changed]) * 10**parameter_log10_rel_step_size

print('pos step')
print(cosmology)

cluster_abundance_pos = N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges)



with open(oup_fname, 'wb') as file:
    print(oup_fname)
    pickle.dump(cluster_abundance_pos, file)



cosmology = fiducial_cosmology.copy()
cosmology[parameter_changed] -= np.abs(cosmology[parameter_changed]) * 10**parameter_log10_rel_step_size

print('neg step')
print(cosmology)



cluster_abundance_neg = N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges)


oup_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/cluster_abundance_fisher_no_mnu_%s_%.4f_neg'%(parameter_changed, parameter_log10_rel_step_size)


with open(oup_fname, 'wb') as file:
    print(oup_fname)
    pickle.dump(cluster_abundance_neg, file)