from aemulusnu_massfunction.emulator import *
from aemulusnu_massfunction.fisher_utils import *
import pickle
import sys

parameter_changed = sys.argv[1]
parameter_log10_rel_step_size = eval(sys.argv[2])

print(len(sys.argv))
sheth_tormen = False
bocquet16 = False
if(len(sys.argv) == 4):
    print('sheth tormen')
    sheth_tormen = True

if(len(sys.argv) == 5):
    print('bocquet16')
    bocquet16 = True



print(parameter_changed, parameter_log10_rel_step_size)

z_bin_edges = [0.2, 0.4, 0.6, 0.8, 1.0]
richness_bin_edges = [20., 30., 45., 60., 300.]

print('fiducial: ')
print(fiducial_cosmology)

cosmology = fiducial_cosmology.copy()
cosmology[parameter_changed] += np.abs(cosmology[parameter_changed]) * 10**parameter_log10_rel_step_size

print('pos step')
print(cosmology)

cluster_abundance_pos = N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges, sheth_tormen=sheth_tormen, bocquet16=bocquet16)



oup_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/cluster_abundance_fisher_%s_%.4f'%(parameter_changed, parameter_log10_rel_step_size)

if(sheth_tormen):
    oup_fname += '_st'
if(bocquet16):
    oup_fname += '_b16'


with open(oup_fname, 'wb') as file:
    print(oup_fname)
    pickle.dump(cluster_abundance_pos, file)



cosmology = fiducial_cosmology.copy()
cosmology[parameter_changed] -= np.abs(cosmology[parameter_changed]) * 10**parameter_log10_rel_step_size

print('neg step')
print(cosmology)



cluster_abundance_neg = N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges, sheth_tormen=sheth_tormen, bocquet16=bocquet16)


oup_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/cluster_abundance_fisher_%s_%.4f_neg'%(parameter_changed, parameter_log10_rel_step_size)

if(sheth_tormen):
    oup_fname += '_st'
if(bocquet16):
    oup_fname += '_b16'


with open(oup_fname, 'wb') as file:
    print(oup_fname)
    pickle.dump(cluster_abundance_neg, file)
