
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
hnn_core_root = op.join(op.dirname(hnn_core.__file__))
emp_dpl = read_dipole(op.join(hnn_core_root, 'yes_trial_S1_ERP_all_avg.txt'))
#emp_dpl = read_dipole('S1_SupraT.txt')

###############################################################################
# Let us first create our default network and visualize the cells
# inside it.
net = L6_model(connect_layer_6=True)
net.plot_cells()
fig = plt.figure(figsize=(6, 6), constrained_layout=True)
for cell_type_idx, cell_type in enumerate(net.cell_types):
    ax = fig.add_subplot(1, len(net.cell_types), cell_type_idx + 1,
                         projection='3d')
    net.cell_types[cell_type].plot_morphology(ax=ax, show=False)
plt.show()

###############################################################################
# Define hyperparameters of repetitive drive sequence
reps = 4
rep_interval = 250.  # in ms; 8 Hz
rep_duration = 170.

tstop = reps * max(rep_interval, rep_duration)
rep_start_times = np.arange(0, tstop, rep_interval)

event_seed = 1

###############################################################################
# Add drives
# note: first define syn weights associated with proximal drive that will
# undergo synaptic depletion
weights_ampa_p1 = {'L2_basket': 0.100, 'L2_pyramidal': 0.200,
                   'L5_basket': 0.030, 'L5_pyramidal': 0.008}
weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 0.5,
                   'L5_basket': 0.0002, 'L5_pyramidal': 0.4}
weights_ampa_L6 = {'L6_pyramidal': 0.05}

for rep_idx, rep_time in enumerate(rep_start_times):

    if rep_idx == reps - 1:  # last rep
        depletion_factor = 0.8 ** -(reps - 1)  # return values to original
        conn_seed = 2
    else:
        depletion_factor = 0.8  # attenuate values on this rep
        conn_seed = 1

    # Prox 1: attenuate syn weight at each repetition
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                            'L5_basket': 1., 'L5_pyramidal': 1.}
    # note that all NMDA weights are zero
    net.add_evoked_drive(
        f'evprox1_rep{rep_idx}', mu=rep_time + 34., sigma=2.47,
        numspikes=1, weights_ampa=weights_ampa_p1,
        location='proximal', synaptic_delays=synaptic_delays_prox,
        probability=1.0, conn_seed=conn_seed, event_seed=event_seed)
    # Prox 1 should also target the distal projections of L6 Pyr
    net.add_evoked_drive(
        f'evprox1_distL6_rep{rep_idx}', mu=rep_time + 34., sigma=2.47,
        numspikes=1, weights_ampa=weights_ampa_L6,
        location='distal', synaptic_delays=0.1,
        probability=0.13, conn_seed=conn_seed, event_seed=event_seed)

    # Dist 1
    weights_ampa_d1 = {'L2_basket': 0.006, 'L2_pyramidal': 0.100,
                       'L5_pyramidal': 0.100}
    weights_nmda_d1 = {'L2_basket': 0.004, 'L2_pyramidal': 0.003,
                       'L5_pyramidal': 0.080}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                          'L5_pyramidal': 0.1}
    net.add_evoked_drive(
        f'evdist1_rep{rep_idx}', mu=rep_time + 65, sigma=3.85, numspikes=1,
        weights_ampa=weights_ampa_d1, weights_nmda=weights_nmda_d1,
        location='distal', synaptic_delays=synaptic_delays_d1,
        probability=1.0, conn_seed=conn_seed, event_seed=event_seed)

    # Prox 2 NB: only AMPA weights differ from first
    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
        f'evprox2_rep{rep_idx}', mu=rep_time + 130, sigma=10.,
        numspikes=1, weights_ampa=weights_ampa_p2,
        location='proximal', synaptic_delays=synaptic_delays_prox,
        probability=1.0, conn_seed=conn_seed, event_seed=event_seed)
    # Prox 2 should also target the distal projections of L6 Pyr
    net.add_evoked_drive(
        f'evprox2_distL6_rep{rep_idx}', mu=rep_time + 130, sigma=10.,
        numspikes=1, weights_ampa=weights_ampa_L6,
        location='distal', synaptic_delays=0.1,
        probability=0.13, conn_seed=conn_seed, event_seed=event_seed)

    # simulate synaptic depletion
    weights_ampa_p1 = {key: weights_ampa_p1[key] * 0.8 for key in
                       weights_ampa_p1.keys()}
    weights_ampa_p2 = {key: weights_ampa_p2[key] * 0.8 for key in
                       weights_ampa_p2.keys()}
    weights_ampa_L6 = {key: weights_ampa_L6[key] * 0.8 for key in
                       weights_ampa_L6.keys()}
###############################################################################
# Now let's simulate the dipole
with MPIBackend(n_procs=6):
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
#net.cell_response.plot_spikes_hist(ax=axes[3], n_bins=200,
#                                   spike_types=['L2_basket', 'L2_pyramidal',
#                                                'L5_basket', 'L5_pyramidal',
#                                                'L6_basket', 'L6_pyramidal'],
#                                   show=False)

###############################################################################
# If you want to analyze how the different cortical layers contribute to
# different net waveform features, then instead of passing ``'agg'`` to
# ``layer``, you can provide a list of layers to be visualized and optionally
# a list of axes to ``ax`` to visualize the dipole moments separately.
plot_dipole(dpls, average=False, layer=['L2', 'L5', 'L6', 'agg'], show=False)
plt.show()