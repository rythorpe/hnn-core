"""Simulate repetition suppression + deviance detection in L6 model."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>


import os.path as op
from urllib.request import urlretrieve

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

from ipywidgets import interact, IntSlider

import hnn_core
from hnn_core import read_dipole
from hnn_core.network import pick_connection
from hnn_core.network_models import L6_model
from hnn_core.viz import plot_dipole, NetworkPlotter, plot_cell_connectivity
from optimization_lib import (cell_groups, layertype_to_grouptype,
                              poiss_drive_params, sim_net_baseline)

###############################################################################
# Read in empirical data to compare to simulated data
data_url = ('https://raw.githubusercontent.com/jonescompneurolab/hnn/master/'
            'data/MEG_detection_data/S1_SupraT.txt')
# urlretrieve(data_url, 'S1_SupraT.txt')
# hnn_core_root = op.join(op.dirname(hnn_core.__file__))
# emp_dpl = read_dipole('yes_trial_S1_ERP_all_avg.txt')
# emp_dpl = read_dipole('S1_SupraT.txt')


def sim_dev_spiking(dev_magnitude=-1, reps=4, ipsirepr_inhib=1.0, n_trials=1,
                    burn_in_time=300.0, n_procs=10, record_vsec=False,
                    rng=None):

    # Hyperparameters of repetitive drive sequence
    stim_interval = 100.  # in ms; 10 Hz
    rep_duration = 100.  # 170 ms for human M/EEG

    # synaptic depression (fractional decrease between [0, 1])
    syn_depression = 0.08

    # see Constantinople and Bruno (2013) and de Kock et al. (2007) for
    # experimental values re: evoked response peak timing
    # see Sachidhanandam (2013) for a discusson on feedback drive timing
    t_prox = 12.  # time (ms) of the proximal drive relative to stimulus rep
    t_dist = 35.  # time (ms) of the distal drive relative to stimulus rep

    # avg connection probability (across groups 1 and 2) controls the
    # proportion of the total network that gets directly activated through
    # afferent drive (an increase or decrease of this value drives deviance
    # detection)
    # by setting number of targetted units that comprise a deviant and then
    # calculating the proportion of Group 1 to Group 2, we ensure this ratio
    # is maintained across standard (std) and deviant (dev) trials despite
    # rounding to the nearest whole unit when drive cells are assigned via
    # probabilities
    grid_shape = (12, 12)
    n_1_delta = 6  # n_cells from group 1 constituting dev drive change
    n_2_delta = 3  # n_cells from group 1 constituting dev drive change
    dev_delta_prob = 1 / 4
    n_agg_cells = grid_shape[0] * grid_shape[1]
    # proportion of red to blue cells targetted by drive
    prop_1_to_2 = n_1_delta / n_2_delta
    # maybe try 0.15 based on Sachidhanandam (2013)?
    prob_avg = (n_1_delta + n_2_delta) / dev_delta_prob / n_agg_cells
    dev_delta = dev_magnitude * dev_delta_prob

    print(f'Proportion of network driven on std: {prob_avg}')
    print(f'# driven units (Group 1) on std: {n_1_delta / dev_delta_prob}')
    print(f'# driven units (Group 2) on std: {n_2_delta / dev_delta_prob}')

    if rng is None:
        rng = np.random.default_rng()

    # for now, seeds are randomly sampled for each drive
    # event_seed = rng.integers(0, np.iinfo(np.int32).max)
    # conn_seed = rng.integers(0, np.iinfo(np.int32).max)

    ###########################################################################
    # Let us first create our network
    net = L6_model(layer_6_fb=True, rng=rng, grid_shape=grid_shape,
                   ipsirepr_inhib=ipsirepr_inhib)
    net.set_cell_positions(inplane_distance=300.0)

    # before we continue, sample a random integer to serve as seed for the
    # baseline (poisson) drive - this will keep baseline drive constant
    # despite differences between the evoked drive event sampling on dev vs.
    # non-dev sims
    poisson_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    ###########################################################################
    # Add drives
    # note: first define syn weights associated with proximal drive that will
    # undergo synaptic depletion

    # prox drive weights and delays
    weights_ampa_prox = {'L2/3i': 0.0008, 'L2/3e': 0.0015,
                         'L5i': 0.0018, 'L5e': 0.0015, 'L6e': 0.0031}
    synaptic_delays_prox = {'L2/3i': 2.0, 'L2/3e': 3.0,
                            'L5i': 3.0, 'L5e': 4.0, 'L6e': 0.0}
    weights_ampa_dist = {'L2/3i': 0.0, 'L2/3e': 0.0004, 'L5e': 0.0007}
    weights_nmda_dist = {'L2/3i': 0.0, 'L2/3e': 0.0001, 'L5e': 0.00005}
    synaptic_delays_dist = {'L2/3i': 0.1, 'L2/3e': 0.1, 'L5e': 0.1}

    # convert each dictionary to a more granular version with specific cell
    # types for each group (e.g., L2e_1 and L2e_2)
    weights_ampa_prox_group = layertype_to_grouptype(
        weights_ampa_prox, cell_groups)
    synaptic_delays_prox_group = layertype_to_grouptype(
        synaptic_delays_prox, cell_groups)
    weights_ampa_dist_group = layertype_to_grouptype(
        weights_ampa_dist, cell_groups)
    weights_nmda_dist_group = layertype_to_grouptype(
        weights_nmda_dist, cell_groups)
    synaptic_delays_dist_group = layertype_to_grouptype(
        synaptic_delays_dist, cell_groups)

    # set drive rep start times from user-defined parameters
    tstop = burn_in_time + reps * max(stim_interval, rep_duration)
    rep_start_times = np.arange(burn_in_time, tstop, stim_interval)
    drive_times = list()
    drive_strengths = list()

    for rep_idx, rep_time in enumerate(rep_start_times):

        # determine if this is the DEV or STD trial...
        if rep_idx == len(rep_start_times) - 1:  # last rep
            # determines the magnitude and direction of the deviant
            prob_delta = dev_delta
        else:
            prob_delta = 0.0

        # attenuate drive strength (proportion of driven post-synaptic targets)
        # as a function rep #
        df = (1 - syn_depression) ** rep_idx
        w_ampa_prox_depressed = {key: val * df for key, val in
                                 weights_ampa_prox_group.items()}
        # drive_strength = (prob_avg + prob_delta) * depression_factor
        drive_strength_default = prob_avg * (1 + prob_delta)
        drive_strengths.append(drive_strength_default)

        prob_prox = dict()
        for layer_type in synaptic_delays_prox.keys():
            # scale L6 delta to make it more extreme
            # must by an integer number to allow a whole number change in
            # the number of driven cells
            drive_strength = drive_strength_default
            if 'L6' in layer_type:
                L6_delta_dev_fctr = 2
                drive_strength = prob_avg * (1 + (L6_delta_dev_fctr * prob_delta))
                print(f'increasing L6 delta to {L6_delta_dev_fctr * prob_delta} '
                      f'on rep {rep_idx}')
            # group-type 1 (red) will be preferentially targetted
            for group_type in cell_groups[layer_type]:
                if '1' in group_type:
                    prop = prop_1_to_2 * 2 / (prop_1_to_2 + 1)
                elif '2' in group_type:
                    prop = 2 / (prop_1_to_2 + 1)
                else:
                    prop = 1
                # prox drive for this cell group: total conn prob has 3
                # factors, proportion of cells targetted relative to other
                # group, the total avg probability of cells targetted across
                # groups, and the depression factor for this rep
                prob_prox[group_type] = prop * drive_strength

        prob_dist = dict()
        for layer_type in synaptic_delays_dist.keys():
            drive_strength = drive_strength_default
            for group_type in cell_groups[layer_type]:
                if '1' in group_type:
                    prop = prop_1_to_2 * 2 / (prop_1_to_2 + 1)
                elif '2' in group_type:
                    prop = 2 / (prop_1_to_2 + 1)
                else:
                    prop = 1
                prob_dist[group_type] = prop * drive_strength

        # store drive times
        drive_times.append({'prox': rep_time + t_prox,
                            'dist': rep_time + t_dist})

        # prox drive: attenuate conn probability at each repetition
        # note that all NMDA weights are zero
        net.add_evoked_drive(
            f'evprox_rep{rep_idx}', mu=drive_times[rep_idx]['prox'],
            sigma=3.0, numspikes=1, weights_ampa=w_ampa_prox_depressed,
            weights_nmda=None,
            location='proximal', synaptic_delays=synaptic_delays_prox_group,
            space_constant=1e50, probability=prob_prox,
            conn_seed=rng.integers(0, np.iinfo(np.int32).max),
            event_seed=rng.integers(0, np.iinfo(np.int32).max))

        # dist drive
        net.add_evoked_drive(
            f'evdist_rep{rep_idx}', mu=drive_times[rep_idx]['dist'],
            sigma=6.0, numspikes=1, weights_ampa=weights_ampa_dist_group,
            weights_nmda=weights_nmda_dist_group,
            location='distal', synaptic_delays=synaptic_delays_dist_group,
            space_constant=1e50, probability=prob_dist,
            conn_seed=rng.integers(0, np.iinfo(np.int32).max),
            event_seed=rng.integers(0, np.iinfo(np.int32).max))

    ###########################################################################
    # Now let's simulate the dipole
    net, dpls = sim_net_baseline(net, sim_time=tstop,
                                 burn_in_time=burn_in_time,
                                 n_trials=n_trials, n_procs=n_procs,
                                 poiss_params=poiss_drive_params,
                                 record_vsec=record_vsec, rng=poisson_seed)

    # window_len, scaling_factor = 30, 2000
    # for dpl in dpls:
    #     dpl.smooth(window_len).scale(scaling_factor)
    drive_params = {'rep_times': rep_start_times, 'drive_times': drive_times,
                    'drive_strengths': drive_strengths, 'tstop': tstop}
    return net, drive_params


def plot_dev_spiking_v1(net, burn_in_time, rep_start_times, drive_times,
                        drive_strengths, tstop, trial_idx=None):
    """Plot the network spiking response to repetitive + deviant drive."""

    # plt.rcParams.update({'font.size': 10})
    custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
    sns.set_theme(style='ticks', rc=custom_params)
    gridspec = {'width_ratios': [1], 'height_ratios': [1, 1, 1, 2]}
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 4),
                             gridspec_kw=gridspec, constrained_layout=True)

    # plot drive strength
    arrow_height_max = 33
    head_length = arrow_height_max / 5
    head_width = 12.0
    stim_interval = np.unique(np.diff(rep_start_times))
    for rep_idx, rep_time in enumerate(rep_start_times):

        drive_strength = drive_strengths[rep_idx] * 100  # convert to percent
        # determine if this is the DEV or STD trial...
        if rep_idx == len(rep_start_times) - 1:  # last rep
            ec = 'k'
            fc = 'k'
        else:
            ec = 'k'
            fc = 'None'

        # plot arrows for each drive
        axes[0].arrow(drive_times[rep_idx]['prox'], 0, 0,
                      drive_strength,
                      fc=fc, ec=ec, alpha=1., width=5, head_width=head_width,
                      head_length=head_length, length_includes_head=True)
        axes[0].arrow(drive_times[rep_idx]['dist'], drive_strength, 0,
                      -drive_strength,
                      fc=fc, ec=ec, alpha=1., width=5, head_width=head_width,
                      head_length=head_length, length_includes_head=True)
        axes[0].hlines(y=drive_strength, xmin=rep_time,
                       xmax=rep_time + stim_interval, colors='k',
                       linestyle=':')
    axes[0].set_ylim([0, arrow_height_max])
    axes[0].set_yticks([0, arrow_height_max])
    axes[0].set_ylabel('external drive\n(% total\ndriven units)')

    # vertical lines separating reps
    for rep_time in rep_start_times:
        axes[0].axvline(rep_time, c='k')
        axes[1].axvline(rep_time, c='k')
        axes[2].axvline(rep_time, c='k')
        axes[3].axvline(rep_time, c='k')

    # horizontal lines separating layers
    # note that the raster plot shows negative GID values on the y-axis
    raster_y_tick_pos = list()
    for layer in ['L2', 'L5', 'L6']:
        greatest_gid = 0
        for cell_name, gid_range in net.gid_ranges.items():
            if layer in cell_name and (gid := max(gid_range)) > greatest_gid:
                greatest_gid = gid
        raster_y_tick_pos.append(-greatest_gid + 108)
        axes[3].axhline(-greatest_gid, c='k')

    # cell groups are separtated in responders (R) and non-responders (NR)
    spike_types = [{'L2/3e': ['L2e_1', 'L2e_2'],
                    'P': ['L2e_1'], 'NP': ['L2e_2']},
                #    {'L4e': ['evprox']},
                #    {'L5e': ['L5e']},
                   {'L6e': ['L6e_1', 'L6e_2'],
                    'P': ['L6e_1'], 'NP': ['L6e_2']}]
    cell_type_colors = {'L2/3e': 'm', 'P': 'r', 'NP': 'b',
                        # 'L4e': 'gray', 'L5e': 'm',
                        'L6e': 'm'}
    for layer_idx, layer_spike_types in enumerate(spike_types):
        for spike_type, spike_type_groups in layer_spike_types.items():
            if 'L4e' in spike_type:
                # this is spiking activity of the proximal drives
                # count artifical drive cells from only one rep
                # (note that each drive has it's own set of artificial cell
                # gids, so the total artifical cell count is inflated compared
                # to the number of L4 stellate cells they represent)
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
                bin_width=10,
                spike_types={spike_type: spike_type_groups},
                color=cell_type_colors[spike_type],
                rate=rate_factor, sliding_bin=True, fill_between=fill_between,
                trial_idx=trial_idx, show=False)

            # finally, plot a horizontal line at the peak agg. spike rate/rep
            if 'P' not in spike_type:
                sr_times = np.array(spike_rates['times'])
                sr = np.array(spike_rates[spike_type])
                for rep_time in rep_start_times:
                    rep_time_stop = rep_time + stim_interval
                    rep_mask = np.logical_and(sr_times >= rep_time,
                                              sr_times < rep_time_stop)
                    peak = sr[rep_mask].max()
                    axes[layer_idx + 1].hlines(
                        y=peak,
                        xmin=rep_time,
                        xmax=rep_time_stop,
                        colors=cell_type_colors[spike_type],
                        linestyle=':'
                    )

    axes[1].set_ylabel('L2/3\nspikes/s')
    handles, _ = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, ['agg. (eP+eNP)', 'eP', 'eNP'], ncol=3,
                   loc='lower center', bbox_to_anchor=(0.5, 1.0),
                   frameon=False, columnspacing=1,
                   handlelength=0.75, borderaxespad=0.0)
    axes[2].set_ylabel('L6\nspikes/s')
    axes[2].get_legend().remove()
    # fig.supylabel('mean single-unit spikes/s')

    # make ylim consistent
    ylim_max = max([axes[1].get_ylim()[1], axes[2].get_ylim()[1]])
    # round up to the nearest multiple of 5 for aesthetics
    ylim_max = (ylim_max // 5 + 1) * 5
    axes[1].set_ylim([0, ylim_max])
    axes[1].set_yticks([0, ylim_max])
    axes[2].set_ylim([0, ylim_max])
    axes[2].set_yticks([0, ylim_max])

    # axes[3].set_ylabel('')
    # axes[3].set_ylim([0, ylim_max])
    # handles, _ = axes[3].get_legend_handles_labels()
    # axes[3].legend(handles, ['L5e'], ncol=1, loc='lower center',
    #                bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1,
    #                handlelength=0.75, borderaxespad=0.0)
    # axes[4].set_ylabel('')
    # axes[4].set_ylim([0, ylim_max])
    # handles, _ = axes[4].get_legend_handles_labels()
    # axes[4].legend(handles, ['L6e P+NP', 'P', 'NP'], ncol=3,
    #                loc='lower center', bbox_to_anchor=(0.5, 1.0),
    #                frameon=False, columnspacing=1,
    #                handlelength=0.75, borderaxespad=0.0)

    spike_types = {'L2i': ['L2i_1', 'L2i_2'],
                   'L2e_1': ['L2e_1'], 'L2e_2': ['L2e_2'],
                #    'L4': [f'evprox_rep{rep}' for rep in
                #           range(len(rep_start_times))],
                   'L5i': ['L5i'], 'L5e': ['L5e'],
                   'L6i': ['L6i_1', 'L6i_2'],
                   'L6e_1': ['L6e_1'], 'L6e_2': ['L6e_2']}
    spike_type_colors = {'L2e_1': 'r', 'L2e_2': 'b', 'L2i': 'orange',
                        #  'L4': 'gray',
                         'L5e': 'gray', 'L5i': 'orange',
                         'L6e_1': 'r', 'L6e_2': 'b', 'L6i': 'orange'}
    net.cell_response.plot_spikes_raster(ax=axes[3], cell_types=spike_types,
                                         color=spike_type_colors, trial_idx=trial_idx, show=False)
    axes[3].set_facecolor('None')
    axes[3].get_yaxis().set_visible(True)
    axes[3].get_legend().remove()
    axes[3].set_xlim([burn_in_time - 100, tstop])
    xticks = np.arange(burn_in_time - 100, tstop + 1, 100)
    xticks_labels = (xticks - rep_start_times[0]).astype(int).astype(str)
    axes[3].set_xticks(xticks)
    axes[3].set_xticklabels(xticks_labels)
    axes[3].set_xlabel('time (ms)')
    axes[3].set_yticks(raster_y_tick_pos)
    axes[3].tick_params('y', width=0)
    axes[3].set_yticklabels(['L2', 'L5', 'L6'])
    axes[3].set_ylabel('single-unit\nspikes')
    # plot_dipole(dpls, average=False, layer=['L2', 'L5', 'L6', 'agg'],
    #             show=False)
    return fig


def plot_dev_spiking_v2(net, burn_in_time, rep_start_times, drive_times,
                        drive_strengths, tstop, trial_idx=None,
                        return_spike_rates=False):
    """Plot the network spiking response to repetitive + deviant drive."""

    # plt.rcParams.update({'font.size': 10})
    custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
    sns.set_theme(style='ticks', rc=custom_params)
    gridspec = {'width_ratios': [1], 'height_ratios': [1, 2.5, 2.5]}
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(4, 3),
                             gridspec_kw=gridspec, constrained_layout=True)

    # plot drive strength
    arrow_height_max = 33
    head_length = arrow_height_max / 5
    head_width = 12.0
    stim_interval = np.unique(np.diff(rep_start_times))
    for rep_idx, rep_time in enumerate(rep_start_times):

        drive_strength = drive_strengths[rep_idx] * 100  # convert to percent
        # determine if this is the DEV or STD trial...
        if rep_idx == len(rep_start_times) - 1:  # last rep
            fc = 'k'
        else:
            fc = 'None'

        # plot arrows for each drive
        axes[0].arrow(drive_times[rep_idx]['prox'], 0, 0,
                      drive_strength, fc=fc, ec='k', lw=0.5, alpha=1.,
                      width=5, head_width=head_width,
                      head_length=head_length, length_includes_head=True)
        axes[0].arrow(drive_times[rep_idx]['dist'], drive_strength, 0,
                      -drive_strength, fc=fc, ec='k', lw=0.5, alpha=1.,
                      width=5, head_width=head_width,
                      head_length=head_length, length_includes_head=True)
        axes[0].hlines(y=drive_strength, xmin=rep_time,
                       xmax=rep_time + stim_interval, colors='k',
                       linestyle=':')
    axes[0].set_ylim([0, arrow_height_max])
    axes[0].set_yticks([0, int(drive_strengths[0] * 100)])
    axes[0].set_ylabel('external drive\n(% total\ndriven units)')

    # vertical lines separating reps
    for rep_time in rep_start_times:
        for ax in axes:
            ax.axvline(rep_time, c='k')

    spike_types_rast = [{'L2i': ['L2i_1', 'L2i_2'],
                         'L2e_1': ['L2e_1'], 'L2e_2': ['L2e_2']},
                        {'L6i': ['L6i_1', 'L6i_2'],
                         'L6e_1': ['L6e_1'], 'L6e_2': ['L6e_2']}]
    spike_type_colors_rast = [{'L2e_1': 'r', 'L2e_2': 'b', 'L2i': 'orange'},
                              {'L6e_1': 'r', 'L6e_2': 'b', 'L6i': 'orange'}]

    # cell groups are separtated in responders (R) and non-responders (NR)
    spike_types_hist = [{'L2/3e': ['L2e_1', 'L2e_2'],
                         'P': ['L2e_1'], 'NP': ['L2e_2']},
                        {'L6e': ['L6e_1', 'L6e_2'],
                         'P': ['L6e_1'], 'NP': ['L6e_2']}]
    cell_type_colors_hist = {'L2/3e': 'm', 'P': 'r', 'NP': 'b',
                             'L6e': 'm'}
    spike_rates_all = dict()
    mean_dev_peak_rates = dict()
    for layer_idx, layer_spike_types in enumerate(spike_types_hist):

        # first plot spike raster in background
        ax_2 = axes[layer_idx + 1].twinx()
        axes[layer_idx + 1].set_zorder(ax_2.get_zorder() + 1)
        axes[layer_idx + 1].patch.set_alpha(0.5)
        net.cell_response.plot_spikes_raster(
            ax=ax_2,
            cell_types=spike_types_rast[layer_idx],
            color=spike_type_colors_rast[layer_idx],
            trial_idx=trial_idx, show=False
        )
        ax_2.get_legend().remove()

        for spike_type, spike_type_groups in layer_spike_types.items():
            if 'L4e' in spike_type:
                # this is spiking activity of the proximal drives
                # count artifical drive cells from only one rep
                # (note that each drive has it's own set of artificial cell
                # gids, so the total artifical cell count is inflated compared
                # to the number of L4 stellate cells they represent)
                n_cells_of_type = \
                    net.external_drives['evprox_rep0']['n_drive_cells']
            else:
                n_cells_of_type = 0
                for spike_type_group in spike_type_groups:
                    n_cells_of_type += len(net.gid_ranges[spike_type_group])
            rate_factor = 1 / n_cells_of_type

            # compute and plot histogram
            # modified to return spike rates as well as create plot
            _, spike_rates = net.cell_response.plot_spikes_hist(
                ax=axes[layer_idx + 1],
                bin_width=10,
                spike_types={spike_type: spike_type_groups},
                color=cell_type_colors_hist[spike_type],
                rate=rate_factor, sliding_bin=True,
                trial_idx=trial_idx, show=False
            )

            # finally, calculate peak spike rates and
            # plot a horizontal line at the peak agg. spike rate/rep
            sr_times = np.array(spike_rates['times'])
            sr = np.array(spike_rates[spike_type])
            for rep_time in rep_start_times:
                rep_time_stop = rep_time + stim_interval
                rep_mask = np.logical_and(sr_times >= rep_time,
                                          sr_times < rep_time_stop)
                peak = sr[rep_mask].max()
                if 'P' not in spike_type:
                    axes[layer_idx + 1].hlines(
                        y=peak,
                        xmin=rep_time,
                        xmax=rep_time_stop,
                        colors=cell_type_colors_hist[spike_type],
                        linestyle=':'
                    )
                # store peak spike rates on dev
                if rep_time == rep_start_times[-1]:
                    if 'P' not in spike_type:
                        spike_type_name = spike_type
                    else:
                        spike_type_name = spike_type_groups[0]
                    mean_dev_peak_rates[spike_type_name] = peak
                        
            if spike_type != 'L2/3e' and spike_type != 'L6e':
                spike_rates_all[spike_type_groups[0]] = spike_rates[spike_type]
                spike_rates_all['times'] = spike_rates['times']

        # round up upper y-axis tick to the nearest multiple of 5 for
        # aesthetics
        ylim_max = axes[layer_idx + 1].get_ylim()[1]
        if layer_spike_types == 'L2/3e':
            round_tick_to = 1  # try 5 if peaks are bigger
        else:
            round_tick_to = 5
        ylim_max = (ylim_max // round_tick_to + 1) * round_tick_to
        axes[layer_idx + 1].set_ylim([0, ylim_max])
        axes[layer_idx + 1].set_yticks([0, ylim_max])

    axes[1].set_ylabel('L2/3\nspikes/s')
    handles, _ = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, ['agg. (eP+eNP)', 'eP', 'eNP'], ncol=3,
                   loc='lower center', bbox_to_anchor=(0.5, 1.0),
                   frameon=False, columnspacing=1,
                   handlelength=0.75, borderaxespad=0.0)
    axes[2].set_ylabel('L6\nspikes/s')
    axes[2].get_legend().remove()
    # fig.supylabel('mean single-unit spikes/s')

    axes[-1].set_xlim([burn_in_time - 100, tstop])
    xticks = np.arange(burn_in_time - 100, tstop + 1, 100)
    xticks_labels = (xticks - rep_start_times[0]).astype(int).astype(str)
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticks_labels)
    axes[-1].set_xlabel('time (ms)')

    if return_spike_rates is True:
        return fig, spike_rates_all, mean_dev_peak_rates
    else:
        return fig


if __name__ == "__main__":

    dev_magnitude = -1
    n_trials = 5
    rng = np.random.default_rng(1234)
    burn_in_time = 300.0
    n_procs = 10
    record_vsec = False

    net, drive_params = sim_dev_spiking(dev_magnitude=dev_magnitude,
                                        n_trials=n_trials,
                                        burn_in_time=burn_in_time,
                                        n_procs=n_procs,
                                        record_vsec=record_vsec,
                                        rng=rng)

    rep_start_times = drive_params['rep_times']
    drive_times = drive_params['drive_times']
    drive_strengths = drive_params['drive_strengths']
    tstop = drive_params['tstop']

    fig_dev_spiking_v1 = plot_dev_spiking_v1(net,
                                             burn_in_time,
                                             rep_start_times,
                                             drive_times,
                                             drive_strengths,
                                             tstop)

    fig_dev_spiking_v2 = plot_dev_spiking_v2(net,
                                             burn_in_time,
                                             rep_start_times,
                                             drive_times,
                                             drive_strengths,
                                             tstop)

    ###########################################################################
    # Plot 3D Network

    # net.plot_cells()
    #
    # fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    # for cell_type_idx, cell_type in enumerate(net.cell_types):
    #     ax = fig.add_subplot(1, len(net.cell_types), cell_type_idx + 1,
    #                          projection='3d')
    #     net.cell_types[cell_type].plot_morphology(ax=ax, show=False)

    # net_small = L6_model(grid_shape=(4, 4))
    # net_small.set_cell_positions(inplane_distance=300.0)
    # sim_net_baseline(net_small, 30., 10., n_trials=1, n_procs=6,
    #                  poiss_params=poiss_drive_params)
    # net_plot = NetworkPlotter(net_small, voltage_colormap='RdBu_r', vmin=-1,
    #                           vmax=1)
    # # net_plot.update_section_voltages(np.argmin(np.abs(net_plot.times - 317.0))) # noqa
    # net_plot.update_section_voltages(0)

    # conn_idxs_1 = pick_connection(net, src_gids='L2i_1', target_gids='L2e_1',
    #                               loc='soma', receptor='gabaa')
    # conn_idxs_2 = pick_connection(net, src_gids='L2i_2', target_gids='L2e_1',
    #                               loc='soma', receptor='gabaa')
    # conn_idxs = [conn_idxs_1[0], conn_idxs_2[0]]
    # src_gids = list(net.connectivity[conn_idxs[0]]['src_gids'])
    # src_gid = src_gids[10]  # select a gid near the middle
    # fig_conn = plot_cell_connectivity(net, conn_idxs, src_gid,
    #                                   colormap='viridis')

    plt.show()
