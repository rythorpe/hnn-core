"""Simulate repetition suppression + deviance detection in L6 model."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>


import os.path as op
from urllib.request import urlretrieve

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

import hnn_core
from hnn_core import simulate_dipole, MPIBackend, JoblibBackend, read_dipole
from hnn_core.network_models import L6_model
from hnn_core.viz import plot_dipole
from optimization_lib import plot_spikerate_hist

###############################################################################
# Read in empirical data to compare to simulated data
data_url = ('https://raw.githubusercontent.com/jonescompneurolab/hnn/master/'
            'data/MEG_detection_data/S1_SupraT.txt')
#urlretrieve(data_url, 'S1_SupraT.txt')
#hnn_core_root = op.join(op.dirname(hnn_core.__file__))
#emp_dpl = read_dipole('yes_trial_S1_ERP_all_avg.txt')
#emp_dpl = read_dipole('S1_SupraT.txt')

###############################################################################
# Define user parameters
model_grid_shape = (10, 10)
# Hyperparameters of repetitive drive sequence
reps = 4
rep_interval = 100.  # in ms; 10 Hz
rep_duration = 100.  # 170 ms for human M/EEG

syn_depletion_factor = 0.8  # used to simulate successive synaptic depression

t_prox = 34.  # time (ms) of the proximal drive relative to stimulus rep
t_dist = 65.  # time (ms) of the distal drive relative to stimulus rep

event_seed = 1
conn_seed = 1

###############################################################################
# Let us first create our default network and visualize the cells
# inside it.
net = L6_model(connect_layer_6=True, legacy_mode=False,
               grid_shape=model_grid_shape)
#net.plot_cells()
#fig = plt.figure(figsize=(6, 6), constrained_layout=True)
#for cell_type_idx, cell_type in enumerate(net.cell_types):
#    ax = fig.add_subplot(1, len(net.cell_types), cell_type_idx + 1,
#                         projection='3d')
    #net.cell_types[cell_type].plot_morphology(ax=ax, show=False)
#plt.show()

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
        depression_factor = syn_depletion_factor ** rep_idx
    weights_ampa_prox_depr = {key: val * depression_factor
                              for key, val in weights_ampa_prox.items()}
    weights_ampa_L6_depr = {key: val * depression_factor
                            for key, val in weights_ampa_L6.items()}

    # Prox 1: attenuate syn weight at each repetition
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                            'L5_basket': 1., 'L5_pyramidal': 1.}
    # note that all NMDA weights are zero
    net.add_evoked_drive(
        f'evprox_rep{rep_idx}', mu=rep_time + t_prox, sigma=2.47, numspikes=1,
        weights_ampa=weights_ampa_prox_depr, weights_nmda=None,
        location='proximal', synaptic_delays=synaptic_delays_prox,
        probability=1.0, conn_seed=conn_seed, event_seed=event_seed)
    # Hack: prox_1 should also target the distal projections of L6 Pyr
    net.add_evoked_drive(
        f'evprox_rep{rep_idx}_L6', mu=rep_time + t_prox, sigma=2.47,
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
        f'evdist_rep{rep_idx}', mu=rep_time + t_dist, sigma=3.85, numspikes=1,
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
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
gridspec = {'width_ratios': [1], 'height_ratios': [1, 1, 1, 1, 1, 3]}
fig, axes = plt.subplots(6, 1, sharex=True, figsize=(6, 6),
                         gridspec_kw=gridspec, constrained_layout=True)

# plot arrow for each drive
arrow_height = 1.0
head_length = 0.2
head_width = 10.0
for rep_idx, rep_time in enumerate(rep_start_times):
    if rep_idx == reps - 1:  # last rep
        depression_factor = 1
    else:
        depression_factor = syn_depletion_factor ** rep_idx
    axes[0].arrow(rep_time + t_prox, 0, 0, arrow_height * depression_factor,
                  fc='k', ec=None,
                  alpha=1., width=2, head_width=head_width,
                  head_length=head_length, length_includes_head=True)
    axes[0].arrow(rep_time + t_dist, 1, 0, -arrow_height, fc='k', ec=None,
                  alpha=1., width=2, head_width=head_width,
                  head_length=head_length, length_includes_head=True)
axes[0].set_ylim([0, arrow_height])
axes[0].set_yticks([0, arrow_height])
axes[0].set_ylabel('drive\nstrength')

for rep_time in rep_start_times:
    axes[0].axvline(rep_time, c='k')
    axes[1].axvline(rep_time, c='k')
    axes[2].axvline(rep_time, c='k')
    axes[3].axvline(rep_time, c='k')
    axes[4].axvline(rep_time, c='k')
    axes[5].axvline(rep_time, c='w')

spike_types = [['L2_basket', 'L2_pyramidal'],
               [{'L4E': ['evprox']}],
               ['L5_basket', 'L5_pyramidal'],
               ['L6_basket', 'L6_pyramidal']]
for layer_idx, layer_spike_types in enumerate(spike_types):
    for spike_type in layer_spike_types:
        if 'L4E' in spike_type:
            # this is spiking activity of the proximal drives
            # count artifical drive cells from only one rep
            # (note that each drive has it's own set of artificial cell gids,
            # so the total artifical cell count is inflated compared to the
            # number of L4 stellate cells they represent)
            n_cells_of_type = (
                net.external_drives['evprox_rep0']['n_drive_cells'] +
                net.external_drives['evprox_rep0_L6']['n_drive_cells']
            )
        else:
            n_cells_of_type = len(net.gid_ranges[spike_type])
        rate_factor = 1 / n_cells_of_type
        net.cell_response.plot_spikes_hist(ax=axes[layer_idx + 1],
                                           bin_width=10,
                                           spike_types=spike_type,
                                           rate=rate_factor, show=False)

axes[1].set_ylabel('mean spike\nrate (Hz)')
axes[1].set_ylim([0, 110])
handles, _ = axes[1].get_legend_handles_labels()
axes[1].legend(handles, ['L2/3I', 'L2/3E'], ncol=2, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1,
               handlelength=0.75, borderaxespad=0.0)
axes[2].set_ylabel('')
axes[2].set_ylim([0, 110])
handles, _ = axes[2].get_legend_handles_labels()
axes[2].legend(handles, ['L4E (proximal drive)'], ncol=2, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1,
               handlelength=0.75, borderaxespad=0.0)
axes[3].set_ylabel('')
axes[3].set_ylim([0, 110])
handles, _ = axes[3].get_legend_handles_labels()
axes[3].legend(handles, ['L5I', 'L5E'], ncol=2, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1,
               handlelength=0.75, borderaxespad=0.0)
axes[4].set_ylabel('')
axes[4].set_ylim([0, 110])
handles, _ = axes[4].get_legend_handles_labels()
axes[4].legend(handles, ['L6I', 'L6E'], ncol=2, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1,
               handlelength=0.75, borderaxespad=0.0)

net.cell_response.plot_spikes_raster(ax=axes[5], show=False)
axes[5].spines[['right', 'top']].set_visible(True)
axes[5].get_legend().remove()
axes[5].set_xlim([0, tstop])
xticks = np.arange(0, tstop + 1, 50)
xticks_labels = (xticks - rep_start_times[-1]).astype(int).astype(str)
axes[5].set_xticks(xticks)
axes[5].set_xticklabels(xticks_labels)
axes[5].set_xlabel('time (ms)')
#plot_dipole(dpls, average=False, layer=['L2', 'L5', 'L6', 'agg'], show=False)
plt.show()
