from emulator import *
from fisher_utils import *
import pickle
import sys

parameter_changed = sys.argv[1]
parameter_log10_rel_step_size = eval(sys.argv[2])

print(parameter_changed, parameter_log10_rel_step_size)

z_bin_edges = [0.2, 0.4, 0.6, 0.8, 1.0]
richness_bin_edges = [20., 30., 45., 60., 300.]

print('fiducial: ')
print(fiducial_cosmology)

cosmology = fiducial_cosmology.copy()
cosmology[parameter_changed] += np.abs(cosmology[parameter_changed]) * 10**parameter_log10_rel_step_size

print('pos step')
print(cosmology)

cluster_abundance_pos = N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges)



with open('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/cluster_abundance_fisher_%s_%.4f'%(parameter_changed, parameter_log10_rel_step_size), 'wb') as file:
    pickle.dump(cluster_abundance_pos, file)



cosmology = fiducial_cosmology.copy()
cosmology[parameter_changed] -= np.abs(cosmology[parameter_changed]) * 10**parameter_log10_rel_step_size

print('neg step')
print(cosmology)



cluster_abundance_neg = N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges)


with open('/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/cluster_abundance_fisher_%s_%.4f_neg'%(parameter_changed, parameter_log10_rel_step_size), 'wb') as file:
    pickle.dump(cluster_abundance_neg, file)
