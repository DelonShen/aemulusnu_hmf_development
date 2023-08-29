from aemulusnu_massfunction.emulator import *
from aemulusnu_massfunction.fisher_utils import *
import pickle
import sys

parameter_changed = sys.argv[1]
parameter_log10_rel_step_size = eval(sys.argv[2])



print(len(sys.argv))
sheth_tormen = False
bocquet16 = False
tinker08 = False
if(len(sys.argv) == 4):
    if(sys.argv[3] == 'st'):
        print('sheth tormen')
        sheth_tormen = True
    if(sys.argv[3] == 'b16'):
        print('bocquet16')
        bocquet16 = True
    if(sys.argv[3] == 't08'):
        print('tinker08')
        tinker08 = True

    if(sys.argv[3] == 'no_mnu'):
        print('no mnu')
        fiducial_cosmology = {'10^9 As':2.09681,
                              'ns': 0.9652,
                              'H0': 67.37,
                              'w0': -1,
                              'ombh2': 0.02233,
                              'omch2': 0.1198,
                              'nu_mass_ev': 0.00,} #this line changed

        fiducial_cosmo_vals = emulator.get_cosmo_vals(fiducial_cosmology)

        fiducial_ccl_cosmo = None

        fiducial_ccl_cosmo = get_ccl_cosmology(tuple(fiducial_cosmo_vals))


print(parameter_changed, parameter_log10_rel_step_size)


oup_fname = '/oak/stanford/orgs/kipac/users/delon/aemulusnu_massfunction/cluster_abundance_fisher_%s_%.4f'%(parameter_changed, parameter_log10_rel_step_size)

if(sheth_tormen):
    oup_fname += '_st'
if(bocquet16):
    oup_fname += '_b16'

if(fiducial_cosmology['nu_mass_ev'] == 0.00):
    oup_fname += '_no_mnu'

print(oup_fname)





z_bin_edges = [0.2, 0.4, 0.6, 0.8, 1.0]
richness_bin_edges = [20., 30., 45., 60., 300.]

print('fiducial: ')
print(fiducial_cosmology)

cosmology = fiducial_cosmology.copy()
cosmology[parameter_changed] += np.abs(cosmology[parameter_changed]) * 10**parameter_log10_rel_step_size

print('pos step')
print(cosmology)

cluster_abundance_pos = N_in_z_bins_and_richness_bins(cosmology, richness_bin_edges, z_bin_edges, sheth_tormen=sheth_tormen, bocquet16=bocquet16)



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
if(fiducial_cosmology['nu_mass_ev'] == 0.00):
    oup_fname += '_no_mnu'

with open(oup_fname, 'wb') as file:
    print(oup_fname)
    pickle.dump(cluster_abundance_neg, file)
