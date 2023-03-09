"""Optimize local network connectivity for realistic resting spikerates."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hnn_core import simulate_dipole, MPIBackend
from hnn_core.viz import plot_dipole


def get_conn_params(loc_net_connections):
    """Get optimization parameters from Network.connectivity attribute."""
    conn_params = list()
    for conn in loc_net_connections:
        conn_params.append(np.log10(conn['nc_dict']['A_weight']))
        conn_params.append(conn['nc_dict']['lamtha'])
    return conn_params


def set_conn_params(net, conn_params):
    """Set updated Network.connectivity parameters in-place."""

    if len(net.connectivity) != (len(conn_params) / 2):
        raise ValueError('Mismatch between size of input conn_params and '
                         'and connections in Network.connectivity')

    for conn_idx, conn in enumerate(net.connectivity):
        conn['nc_dict']['A_weight'] = conn_params[conn_idx * 2]
        conn['nc_dict']['lamtha'] = conn_params[conn_idx * 2 + 1]


def plot_net_response(dpls, net, sim_time):
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(6, 6),
                             constrained_layout=True)

    #window_len, scaling_factor = 30, 2000
    #for dpl in dpls:
    #    dpl.smooth(window_len).scale(scaling_factor)

    net.cell_response.plot_spikes_hist(ax=axes[0], n_bins=sim_time, show=False)
    plot_dipole(dpls, ax=axes[1:5], layer=['L2', 'L5', 'L6', 'agg'],
                show=False)
    net.cell_response.plot_spikes_raster(ax=axes[5], show=False)
    return fig


def plot_spiking_profiles(net, sim_time, burn_in_time):
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(6, 6),
                             constrained_layout=True)

    for cell_type_idx, cell_type in enumerate(net.cell_types):
        spike_gids = np.array(net.cell_response.spike_gids[0])  # only 1 trial
        spike_times = np.array(net.cell_response.spike_times[0])  # same
        n_cells = len(net.gid_ranges[cell_type])
        spike_rates = np.zeros((n_cells,))
        for gid_idx, gid in enumerate(net.gid_ranges[cell_type]):
            gids_after_burn_in = np.array(spike_gids)[spike_times >
                                                      burn_in_time]
            n_spikes = np.sum(gids_after_burn_in == gid)
            spike_rates[gid_idx] = (n_spikes /
                                    ((sim_time - burn_in_time) * 1e-3))
        sns.histplot(data=spike_rates,
                     ax=axes[cell_type_idx // 2, cell_type_idx % 2],
                     stat='probability')
        axes[cell_type_idx // 2, cell_type_idx % 2].set_ylim([0, 1])
        axes[cell_type_idx // 2, cell_type_idx % 2].set_yticks([0, 0.5, 1])
        axes[cell_type_idx // 2, cell_type_idx % 2].set_ylabel('')
        axes[cell_type_idx // 2, cell_type_idx % 2].set_title(cell_type)
    axes[0, 0].set_ylabel('probability')
    axes[-1, 0].set_xlabel('spike rate (Hz)')
    return fig


def simulate_network(net, sim_time, burn_in_time, n_procs=6,
                     poiss_params=None, conn_params=None, clear_conn=False):
    """Update network with sampled params and run simulation."""
    net = net.copy()

    if conn_params is not None:
        print('resetting network connectivity')
        # transform synaptic weight params from log10->R scale
        conn_params_transformed = np.array(conn_params.copy())
        # every other element is a synaptic weight param
        conn_params_transformed[::2] = 10 ** conn_params_transformed[::2]
        # update local network connections with new params
        set_conn_params(net, conn_params_transformed)

    # when optimizing cell excitability under poisson drive, it's nice to use
    # a disconnected network
    if clear_conn is True:
        print("simulating disconnected network")
        net.clear_connectivity()
    else:
        print("simulating fully-connected network")

    if poiss_params is not None:
        # add the same poisson drive as before
        cell_types = ['L2_basket', 'L2_pyramidal',
                      'L5_basket', 'L5_pyramidal',
                      'L6_basket', 'L6_pyramidal']
        poiss_weights = {cell_type: weight for cell_type, weight in
                         zip(cell_types, poiss_params[:-1])}
        poiss_rate = poiss_params[-1]
        seed = int(np.random.random() * 1e3)
        # add poisson drive with near-uniform spatial spread
        net.add_poisson_drive(name='poisson_drive',
                              rate_constant=poiss_rate,
                              location='soma',
                              n_drive_cells='n_cells',
                              cell_specific=True,
                              weights_ampa=poiss_weights,
                              synaptic_delays=0.0,
                              space_constant=1e14,
                              probability=1.0,
                              event_seed=seed)

    with MPIBackend(n_procs=n_procs):
        dpls = simulate_dipole(net, tstop=sim_time, n_trials=1,
                               baseline_win=[burn_in_time, sim_time])

    return net, dpls


def err_disconn_spike_rate(net, sim_time, burn_in_time,
                           avg_expected_spike_rates):
    """Cost function for matching simulated vs expected avg spike rates.

    Used for optimizing cell excitability under poisson drive.
    """
    avg_spike_rates = net.cell_response.mean_rates(tstart=burn_in_time,
                                                   tstop=sim_time,
                                                   gid_ranges=net.gid_ranges)

    spike_rate_diffs = list()
    for cell_type in avg_expected_spike_rates.keys():
        spike_rate_diffs.append(avg_expected_spike_rates[cell_type] -
                                avg_spike_rates[cell_type])

    return np.linalg.norm(spike_rate_diffs)


def opt_baseline_spike_rates(opt_params, net, sim_params):
    """Function to minimize during optimization: err in baseline spikerates."""
    opt_params = np.array(opt_params)
    sim_time = sim_params['sim_time']
    burn_in_time = sim_params['burn_in_time']
    n_procs = sim_params['n_procs']

    # taken from Reyes-Puerta 2015 and De Kock 2007
    # see Constantinople and Bruno 2013 for laminar difference in E-cell
    # excitability and proportion of connected pairs
    target_avg_spike_rates = {'L2_basket': 0.8,
                              'L2_pyramidal': 0.3,
                              'L5_basket': 2.4,  # L5A + L5B avg
                              'L5_pyramidal': 1.4,  # L5A + L5B avg
                              'L6_basket': 1.3,  # estimated; Reyes-Puerta 2015
                              'L6_pyramidal': 0.5}  # from De Kock 2007

    #net, dpls = simulate_network(conn_params, poiss_params, clear_conn=False)
    #err = err_disconn_spike_rate(net,
    #                             target_avg_spike_rates,
    #                             burn_in_time=burn_in_time,
    #                             sim_time=sim_time)

    # avg rates in unconn network should be a bit less
    # try 20% of the avg rates in a fully connected network
    target_avg_spike_rates_unconn = {cell: rate * 0.33 for cell, rate in
                                     target_avg_spike_rates.items()}
    # for now we'll make them uniform: 10% of cells will fire per second
    #target_avg_spike_rates_unconn = {cell: 0.1 for cell in
    #                                 target_avg_spike_rates.keys()}

    # convert weight param from back from log_10 scale
    poiss_params = np.append(10 ** opt_params[:-1], opt_params[-1])
    net_disconn, dpls_disconn = simulate_network(net, sim_time, burn_in_time,
                                                 n_procs,
                                                 poiss_params=poiss_params,
                                                 conn_params=None,
                                                 clear_conn=True,)
    # note: pass in global variables "burn_in_time" and "sim_time"
    err = err_disconn_spike_rate(net_disconn, sim_time, burn_in_time,
                                 target_avg_spike_rates_unconn)
    return err
