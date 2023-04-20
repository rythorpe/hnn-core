"""Simulate repetition suppression + deviance detection in L6 model."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>


import os.path as op
from urllib.request import urlretrieve

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, MPIBackend, JoblibBackend, read_dipole
from hnn_core.network_models import L6_model, calcium_model
from hnn_core.viz import plot_dipole

data_url = ('https://raw.githubusercontent.com/jonescompneurolab/hnn/master/'
            'data/MEG_detection_data/S1_SupraT.txt')
#urlretrieve(data_url, 'S1_SupraT.txt')
#hnn_core_root = op.join(op.dirname(hnn_core.__file__))
emp_dpl = read_dipole('yes_trial_S1_ERP_all_avg.txt')
#emp_dpl = read_dipole('S1_SupraT.txt')

###############################################################################
# Define user parameters
model_grid_shape = (10, 10)
# Hyperparameters of repetitive drive sequence
reps = 4
rep_interval = 100.  # in ms; 10 Hz
rep_duration = 100.  # 170 ms for human M/EEG

syn_depletion_factor = 0.8  # used to simulate successive synaptic depression

event_seed = 1
conn_seed = 1

###############################################################################
# Let us first create our default network and visualize the cells
# inside it.
net = L6_model(connect_layer_6=True, legacy_mode=False,
               grid_shape=model_grid_shape)
net.plot_cells()
fig = plt.figure(figsize=(6, 6), constrained_layout=True)
for cell_type_idx, cell_type in enumerate(net.cell_types):
    ax = fig.add_subplot(1, len(net.cell_types), cell_type_idx + 1,
                         projection='3d')
    net.cell_types[cell_type].plot_morphology(ax=ax, show=False)
plt.show()

###############################################################################
# Add drives
# note: first define syn weights associated with proximal drive that will
# undergo synaptic depletion
weights_ampa_prox = {'L2_basket': 0.100, 'L2_pyramidal': 0.200,
                     'L5_basket': 0.030, 'L5_pyramidal': 0.008}
weights_ampa_L6 = {'L6_pyramidal': 0.01}

# set drive rep start times from user-defined parameters
tstop = reps * max(rep_interval, rep_duration)
rep_start_times = np.arange(0, tstop, rep_interval)

for rep_idx, rep_time in enumerate(rep_start_times):

    # downscale syn weights for each successive prox drive, except on last rep
    if rep_idx == reps - 1:  # last rep
        # no depression
        depression_factor = 1
    else:
        # attenuate syn weight values as a function of # of reps
        depression_factor = syn_depletion_factor ** (rep_idx + 1)
    weights_ampa_prox_depr = {key: val * depression_factor
                              for key, val in weights_ampa_prox.items()}
    weights_ampa_L6_depr = {key: val * depression_factor
                            for key, val in weights_ampa_L6.items()}

    # Prox 1: attenuate syn weight at each repetition
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                            'L5_basket': 1., 'L5_pyramidal': 1.}
    # note that all NMDA weights are zero
    net.add_evoked_drive(
        f'evprox_rep{rep_idx}', mu=rep_time + 34., sigma=2.47, numspikes=1,
        weights_ampa=weights_ampa_prox_depr, weights_nmda=None,
        location='proximal', synaptic_delays=synaptic_delays_prox,
        probability=1.0, conn_seed=conn_seed, event_seed=event_seed)
    # Hack: prox_1 should also target the distal projections of L6 Pyr
    net.add_evoked_drive(
        f'evprox_rep{rep_idx}_L6', mu=rep_time + 34., sigma=2.47,
        numspikes=1, weights_ampa=weights_ampa_L6_depr,
        location='distal', synaptic_delays=0.1,
        probability=0.33, conn_seed=conn_seed, event_seed=event_seed)

    # Dist 1
    weights_ampa_dist = {'L2_basket': 0.006, 'L2_pyramidal': 0.100,
                         'L5_pyramidal': 0.100}
    weights_nmda_dist = {'L2_basket': 0.004, 'L2_pyramidal': 0.003,
                         'L5_pyramidal': 0.080}
    synaptic_delays_dist = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                            'L5_pyramidal': 0.1}
    net.add_evoked_drive(
        f'evdist_rep{rep_idx}', mu=rep_time + 65, sigma=3.85, numspikes=1,
        weights_ampa=weights_ampa_dist, weights_nmda=weights_nmda_dist,
        location='distal', synaptic_delays=synaptic_delays_dist,
        probability=1.0, conn_seed=conn_seed, event_seed=event_seed)

###############################################################################
# Now let's simulate the dipole
with MPIBackend(n_procs=10):
#with JoblibBackend(n_jobs=1):
    dpls = simulate_dipole(net, tstop=tstop, n_trials=1)

window_len, scaling_factor = 30, 2000
for dpl in dpls:
    dpl.smooth(window_len).scale(scaling_factor)

###############################################################################
# Plot the amplitudes of the simulated aggregate dipole moments over time
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
net.cell_response.plot_spikes_hist(ax=axes[0], n_bins=200,
                                   spike_types=['evprox', 'evdist'],
                                   show=False)
for rep_time in rep_start_times:
    axes[0].axvline(rep_time, c='k')
    axes[1].axvline(rep_time, c='k')
    axes[2].axvline(rep_time, c='w')
plot_dipole(dpls, ax=axes[1], layer='agg', show=False)
emp_dpl.plot(ax=axes[1], color='tab:purple', show=False)
net.cell_response.plot_spikes_raster(ax=axes[2], show=False)
plot_dipole(dpls, average=False, layer=['L2', 'L5', 'L6', 'agg'], show=False)
plt.show()
