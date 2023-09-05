"""Simulate repetition suppression + deviance detection in L6 model."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>


import os.path as op
from urllib.request import urlretrieve

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

import hnn_core
from hnn_core import read_dipole
from hnn_core.network_models import L6_model
from hnn_core.viz import plot_dipole
from optimization_lib import (cell_groups, special_groups,
                              layertype_to_grouptype, poiss_drive_params,
                              simulate_network)

###############################################################################
# Read in empirical data to compare to simulated data
data_url = ('https://raw.githubusercontent.com/jonescompneurolab/hnn/master/'
            'data/MEG_detection_data/S1_SupraT.txt')
# urlretrieve(data_url, 'S1_SupraT.txt')
# hnn_core_root = op.join(op.dirname(hnn_core.__file__))
# emp_dpl = read_dipole('yes_trial_S1_ERP_all_avg.txt')
# emp_dpl = read_dipole('S1_SupraT.txt')

###############################################################################
# Define user parameters
# general sim parameters
n_procs = 10
burn_in_time = 300.0

# Hyperparameters of repetitive drive sequence
reps = 4
stim_interval = 100.  # in ms; 10 Hz
rep_duration = 100.  # 170 ms for human M/EEG

syn_depletion_factor = 0.9  # used to simulate successive synaptic depression

# see Constantinople and Bruno (2013) for experimental values
# see Sachidhanandam (2013) for a discusson on feedback drive timing
t_prox = 12.  # time (ms) of the proximal drive relative to stimulus rep
t_dist = 40.  # time (ms) of the distal drive relative to stimulus rep

# connection probability controls the proportion of the circuit gets directly
# activated through afferent drive (increase or decrease this value on the
# deviant rep)
prox_conn_prob_std = 0.20  # maybe try 0.15 based on Sachidhanandam (2013)?
prox_conn_prob_dev = 0.10  # THIS EVOKES THE DEVIANT!!!!
dist_conn_prob_std = 0.20
dist_conn_prob_dev = 0.20


event_seed = 1
conn_seed = 1

###############################################################################
# Let us first create our default network and visualize the cells
# inside it.
net = L6_model(connect_layer_6=True)
# net.plot_cells()
# fig = plt.figure(figsize=(6, 6), constrained_layout=True)
# for cell_type_idx, cell_type in enumerate(net.cell_types):
#    ax = fig.add_subplot(1, len(net.cell_types), cell_type_idx + 1,
#                         projection='3d')
#     net.cell_types[cell_type].plot_morphology(ax=ax, show=False)
# plt.show()

###############################################################################
# Add drives
# note: first define syn weights associated with proximal drive that will
# undergo synaptic depletion

# prox drive weights and delays
weights_ampa_prox = layertype_to_grouptype(
    {'L2/3i': 0.0, 'L2/3e': 0.009, 'L5i': 0.0, 'L5e': 0.0025, 'L6e': 0.01},
    cell_groups)
synaptic_delays_prox = layertype_to_grouptype(
    {'L2/3i': 0.1, 'L2/3e': 0.1, 'L5i': 1., 'L5e': 1., 'L6e': 0.1},
    cell_groups)
weights_ampa_dist = layertype_to_grouptype(
    {'L2/3i': 0.0, 'L2/3e': 0.009, 'L5e': 0.0023}, cell_groups)
weights_nmda_dist = layertype_to_grouptype(
    {'L2/3i': 0.0, 'L2/3e': 0.0, 'L5e': 0.0}, cell_groups)
synaptic_delays_dist = layertype_to_grouptype(
    {'L2/3i': 0.1, 'L2/3e': 0.1, 'L5e': 0.1}, cell_groups)

# set drive rep start times from user-defined parameters
tstop = burn_in_time + reps * max(stim_interval, rep_duration)
rep_start_times = np.arange(burn_in_time, tstop, stim_interval)

for rep_idx, rep_time in enumerate(rep_start_times):

    # downscale syn weights for each successive prox drive
    # attenuate syn weight values as a function of # of reps
    depression_factor = syn_depletion_factor ** rep_idx

    # determine if this is the DEV or STD trial...
    if rep_idx == len(rep_start_times) - 1:  # last rep
        # THIS EVOKES THE DEVIANT!!!!
        prox_strength = prox_conn_prob_dev
        dist_strength = dist_conn_prob_dev
    else:
        prox_strength = prox_conn_prob_std
        dist_strength = dist_conn_prob_std

    # weights_ampa_prox_depr = {key: val * depression_factor
    #                           for key, val in weights_ampa_prox.items()}
    # weights_ampa_L6_depr = {key: val * depression_factor
    #                         for key, val in weights_ampa_L6.items()}

    # prox drive: attenuate conn probability at each repetition
    # note that all NMDA weights are zero
    net.add_evoked_drive(
        f'evprox_rep{rep_idx}', mu=rep_time + t_prox, sigma=2.47, numspikes=1,
        weights_ampa=weights_ampa_prox, weights_nmda=None,
        location='proximal', synaptic_delays=synaptic_delays_prox,
        probability=prox_strength * depression_factor,
        conn_seed=conn_seed, event_seed=event_seed)

    # dist drive
    net.add_evoked_drive(
        f'evdist_rep{rep_idx}', mu=rep_time + t_dist, sigma=3.85, numspikes=1,
        weights_ampa=weights_ampa_dist, weights_nmda=weights_nmda_dist,
        location='distal', synaptic_delays=synaptic_delays_dist,
        probability=dist_strength,
        conn_seed=conn_seed, event_seed=event_seed)

###############################################################################
# Now let's simulate the dipole
net, dpls = simulate_network(net, sim_time=tstop, burn_in_time=burn_in_time,
                             n_trials=1, n_procs=n_procs,
                             poiss_params=poiss_drive_params)
# with MPIBackend(n_procs=10):
#     dpls = simulate_dipole(net, tstop=tstop, n_trials=1)

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

# plot drive strength
arrow_height_max = 1.0
head_length = 0.2
head_width = 12.0
for rep_idx, rep_time in enumerate(rep_start_times):

    # determine if this is the DEV or STD trial...
    if rep_idx == len(rep_start_times) - 1:  # last rep
        # THIS EVOKES THE DEVIANT!!!!
        prox_strength = prox_conn_prob_dev
        dist_strength = dist_conn_prob_dev
    else:
        prox_strength = prox_conn_prob_std
        dist_strength = dist_conn_prob_std

    # plot arrows for each drive
    axes[0].arrow(rep_time + t_prox, 0, 0, arrow_height_max * prox_strength,
                  fc='k', ec=None, alpha=1., width=5, head_width=head_width,
                  head_length=head_length, length_includes_head=True)
    axes[0].arrow(rep_time + t_dist, dist_strength, 0, -dist_strength,
                  fc='k', ec=None, alpha=1., width=5, head_width=head_width,
                  head_length=head_length, length_includes_head=True)
axes[0].set_ylim([0, arrow_height_max])
axes[0].set_yticks([0, arrow_height_max])
axes[0].set_ylabel('drive\nstrength')

# vertical lines separating reps
for rep_time in rep_start_times:
    axes[0].axvline(rep_time, c='k')
    axes[1].axvline(rep_time, c='k')
    axes[2].axvline(rep_time, c='k')
    axes[3].axvline(rep_time, c='k')
    axes[4].axvline(rep_time, c='k')
    axes[5].axvline(rep_time, c='w', alpha=0.5)

# horizontal lines separating layers
layer_separators = list()
for layer in ['L2', 'L5']:
    greatest_gid = 0
    for cell_name, gid_range in net.gid_ranges.items():
        if layer in cell_name and (gid := max(gid_range)) > greatest_gid:
            greatest_gid = gid
    axes[5].axhline(-greatest_gid, c='w', alpha=0.5)

# cell groups are separtated in responders (R) and non-responders (NR)
spike_types = [{'L2/3e': ['L2e_1', 'L2e_2'], 'P': ['L2e_1'], 'NP': ['L2e_2']},
               {'L4e': ['evprox']},
               {'L5e': ['L5e']},
               {'L6e': ['L6e_1', 'L6e_2'], 'P': ['L6e_1'], 'NP': ['L6e_2']}]
cell_type_colors = {'L2/3e': 'm', 'P': 'r', 'NP': 'b',
                    'L4e': 'gray', 'L5e': 'm',
                    'L6e': 'm'}
for layer_idx, layer_spike_types in enumerate(spike_types):
    for spike_type, spike_type_groups in layer_spike_types.items():
        if 'L4e' in spike_type:
            # this is spiking activity of the proximal drives
            # count artifical drive cells from only one rep
            # (note that each drive has it's own set of artificial cell gids,
            # so the total artifical cell count is inflated compared to the
            # number of L4 stellate cells they represent)
            n_cells_of_type = \
                net.external_drives['evprox_rep0']['n_drive_cells']
        else:
            n_cells_of_type = 0
            for spike_type_group in spike_type_groups:
                n_cells_of_type += len(net.gid_ranges[spike_type_group])
        rate_factor = 1 / n_cells_of_type

        # compute and plot histogram
        # fill area under curve only for aggregate spike rates
        fill_between = True
        if 'P' in spike_type:
            fill_between = False
        # modified to return spike rates as well as create plot
        _, spike_rates = net.cell_response.plot_spikes_hist(
            ax=axes[layer_idx + 1],
            bin_width=5,
            spike_types={spike_type: spike_type_groups},
            color=cell_type_colors[spike_type],
            rate=rate_factor, sliding_bin=True, fill_between=fill_between,
            show=False)

        # finally, plot a horizontal line at the peak agg. spike rate per/rep
        if 'P' not in spike_type:
            sr_times = np.array(spike_rates['times'])
            sr = np.array(spike_rates[spike_type])
            for rep_time in rep_start_times:
                rep_time_stop = rep_time + stim_interval
                rep_mask = np.logical_and(sr_times >= rep_time,
                                          sr_times < rep_time_stop)
                peak = sr[rep_mask].max()
                axes[layer_idx + 1].hlines(y=peak, xmin=rep_time,
                                           xmax=rep_time_stop,
                                           colors=cell_type_colors[spike_type],
                                           linestyle=':')

axes[1].set_ylabel('mean single-unit\nspikes/s')
axes[1].set_ylim([0, 150])
handles, _ = axes[1].get_legend_handles_labels()
axes[1].legend(handles, ['L2/3e R+NR', 'P', 'NP'], ncol=3, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1,
               handlelength=0.75, borderaxespad=0.0)
axes[2].set_ylabel('')
axes[2].set_ylim([0, 150])
handles, _ = axes[2].get_legend_handles_labels()
axes[2].legend(handles, ['L4e (proximal drive)'], ncol=1, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1,
               handlelength=0.75, borderaxespad=0.0)
axes[3].set_ylabel('')
axes[3].set_ylim([0, 150])
handles, _ = axes[3].get_legend_handles_labels()
axes[3].legend(handles, ['L5e'], ncol=1, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1,
               handlelength=0.75, borderaxespad=0.0)
axes[4].set_ylabel('')
axes[4].set_ylim([0, 150])
handles, _ = axes[4].get_legend_handles_labels()
axes[4].legend(handles, ['L6e R+NR', 'P', 'NP'], ncol=3, loc='lower center',
               bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1,
               handlelength=0.75, borderaxespad=0.0)

spike_types = {'L2i': ['L2i_1', 'L2i_2'],
               'L2e_1': ['L2e_1'], 'L2e_2': ['L2e_2'],
               'L5i': ['L5i'], 'L5e': ['L5e'],
               'L6i': ['L6i_1', 'L6i_2'],
               'L6e_1': ['L6e_1'], 'L6e_2': ['L6e_2'],
               'L6i_cross1': ['L6i_cross1'], 'L6i_cross2': ['L6i_cross2']}
spike_type_colors = {'L2e_1': 'r', 'L2e_2': 'b', 'L2i': 'orange',
                     'L5e': 'm', 'L5i': 'orange',
                     'L6e_1': 'r', 'L6e_2': 'b', 'L6i': 'orange',
                     'L6i_cross1': 'orange', 'L6i_cross2': 'orange'}
net.cell_response.plot_spikes_raster(ax=axes[5], cell_types=spike_types,
                                     color=spike_type_colors, show=False)
axes[5].spines[['right', 'top']].set_visible(True)
axes[5].get_legend().remove()
axes[5].set_xlim([burn_in_time - 100, tstop])
xticks = np.arange(burn_in_time - 100, tstop + 1, 50)
xticks_labels = (xticks - rep_start_times[0]).astype(int).astype(str)
axes[5].set_xticks(xticks)
axes[5].set_xticklabels(xticks_labels)
axes[5].set_xlabel('time (ms)')
axes[5].set_ylabel('cell #')
# plot_dipole(dpls, average=False, layer=['L2', 'L5', 'L6', 'agg'], show=False)
plt.show()
