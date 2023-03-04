"""Optimize local network connectivity for realistic resting spikerates."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skopt import gp_minimize
from skopt.plots import plot_gaussian_process, plot_convergence

from hnn_core import simulate_dipole, MPIBackend
from hnn_core.network_models import L6_model
from hnn_core.viz import plot_dipole


###############################################################################
# user parameters
poiss_weight = 7e-4
# 1 kHz as in Billeh et al. 2020 is too fast for this size of network
# decreasing to 10 Hz seems to allow for random single-cell events in a
# disconnected network
poiss_rate = 1e1
# note that basket cells and pyramidal cells require different amounts of AMPA
# excitatory current in order to drive a spike
#poiss_weights = {'L2_basket': 6e-4, 'L2_pyramidal': 8e-4,
#                 'L5_basket': 6e-4, 'L5_pyramidal': 8e-4,
#                 'L6_basket': 6e-4, 'L6_pyramidal': 8e-4}
sim_time = 600  # ms
burn_in_time = 200  # ms

# log_10 scale
min_weight, max_weight = -5., -1.
# real number scale; also applies to poisson rate
min_lamtha, max_lamtha = 1., 100.

# opt parameters
opt_n_total_calls = 50
opt_n_init_points = 25

net_original = L6_model(connect_layer_6=True, legacy_mode=False,
                        grid_shape=(10, 10))

np.random.seed(1)


###############################################################################
# functions
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


def plot_net_response(dpls, net):
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


def simulate_network(poiss_params, conn_params=None, clear_conn=False):
    """Update network with sampled params and run simulation."""
    net = net_original.copy()

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

    # add the same poisson drive as before
    seed = int(np.random.random() * 1e3)
    poiss_weight = 10 ** poiss_params[0]
    poiss_weights = {cell_type: poiss_weight for cell_type in net.cell_types}
    poiss_rate = poiss_params[1]
    net.add_poisson_drive(name='poisson_drive',
                          rate_constant=poiss_rate,
                          location='soma',
                          n_drive_cells='n_cells',
                          cell_specific=True,
                          weights_ampa=poiss_weights,
                          synaptic_delays=0.0,
                          space_constant=1e14,  # near-uniform spatial spread
                          probability=1.0,
                          event_seed=seed)

    with MPIBackend(n_procs=6):
        dpls = simulate_dipole(net, tstop=sim_time, n_trials=1,
                               baseline_win=[burn_in_time, sim_time])

    return net, dpls


def err_disconn_spike_rate(net, avg_expected_spike_rates,
                           burn_in_time, sim_time):
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


def opt_min_func(opt_params):
    """Function to minimize during optimization: err in baseline spikerates."""

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
    target_avg_spike_rates_unconn = {cell: rate * 0.2 for cell, rate in
                                     target_avg_spike_rates.items()}
    # for now we'll make them uniform: 10% of cells will fire per second
    #target_avg_spike_rates_unconn = {cell: 0.1 for cell in
    #                                 target_avg_spike_rates.keys()}

    net_disconn, dpls_disconn = simulate_network(poiss_params=opt_params,
                                                 conn_params=None,
                                                 clear_conn=True,)
    # note: pass in global variables "burn_in_time" and "sim_time"
    err = err_disconn_spike_rate(net_disconn,
                                 target_avg_spike_rates_unconn,
                                 burn_in_time=burn_in_time,
                                 sim_time=sim_time)
    return err


###############################################################################
# get initial params prior to optimization
#opt_params_0 = get_conn_params(net_original.connectivity)
opt_params_0 = [np.log10(poiss_weight), poiss_rate]
opt_params_bounds = np.tile([[min_weight, max_weight],
                             [min_lamtha, max_lamtha]],
                            (len(opt_params_0) // 2, 1))

###############################################################################
# optimize
opt_results = gp_minimize(func=opt_min_func,
                          dimensions=opt_params_bounds,
                          x0=opt_params_0,
                          n_calls=opt_n_total_calls,  # >5**n_params
                          n_initial_points=opt_n_init_points,  # 5**n_params
                          initial_point_generator='lhs',  # sobol; params<40
                          acq_optimizer='sampling',
                          verbose=True,
                          random_state=1234)
opt_params = opt_results.x
print(f'poiss_weight: {10 ** opt_params[0]}')
print(f'poiss_rate: {opt_params[1]}')

###############################################################################
# plot results
plot_convergence(opt_results, ax=None)
#plot_gaussian_process(opt_results)

# pre-optimization
# note: poiss_params expects a weight param in log_10 scale
net_0, dpls_0 = simulate_network(poiss_params=opt_params_0, clear_conn=True)
net_response_fig = plot_net_response(dpls_0, net_0)
sr_profiles_fig = plot_spiking_profiles(net_0, sim_time, burn_in_time)
# post-optimization
net, dpls = simulate_network(poiss_params=opt_params, clear_conn=True)
net_response_fig = plot_net_response(dpls, net)
sr_profiles_fig = plot_spiking_profiles(net, sim_time, burn_in_time)

plt.show()
