"""Optimize local network connectivity for realistic resting spikerates."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hnn_core import simulate_dipole, MPIBackend
from hnn_core.network_models import L6_model
from hnn_core.viz import plot_dipole


###############################################################################
# user parameters
poiss_freq = 1e3  # 1 kHz as in Billeh et al. 2020
poiss_weight = 5e-5
poiss_seed_rng = np.random.default_rng(1)
sim_time = 500  # ms
burn_in_time = 100  # ms

min_weight, max_weight = -5., -1.  # log_10 scale
min_lamtha, max_lamtha = 1., 100.  # real number scale


###############################################################################
# functions
def get_conn_params(loc_net_connections):
    conn_params = list()
    for conn in loc_net_connections:
        conn_params.append(conn['nc_dict']['A_weight'])
        conn_params.append(conn['nc_dict']['lamtha'])
    return np.array(conn_params)


def plot_net_response(dpls, net):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6),
                             constrained_layout=True)

    window_len, scaling_factor = 30, 2000
    for dpl in dpls:
        dpl.smooth(window_len).scale(scaling_factor)

    net.cell_response.plot_spikes_hist(ax=axes[0], n_bins=sim_time, show=False)
    plot_dipole(dpls, ax=axes[1], layer='agg', show=False)
    net.cell_response.plot_spikes_raster(ax=axes[2], show=False)
    return fig


def plot_sr_profiles(net, sim_time, burn_in_time):
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


def set_conn_parameters_and_simulate(conn_params):
    net = net.copy()
    cell_types = list(net.cell_types.keys())
    seed = int(poiss_seed_rng.random() * 1e3)
    net.add_poisson_drive(name='baseline_drive',
                          rate_constant=poiss_freq,
                          location='soma',
                          n_drive_cells='n_cells',
                          cell_specific=True,
                          weights_ampa={cell: poiss_weight for cell in cell_types},
                          synaptic_delays=0.0,
                          space_constant=1e14,  # diminish impact of space constant
                          probability=1.0,
                          event_seed=seed)

    with MPIBackend(n_procs=10):
        dpls = simulate_dipole(net, tstop=sim_time, n_trials=1)


###############################################################################
# get initial params prior to optimization
net = L6_model(connect_layer_6=True, legacy_mode=False, grid_shape=(10, 10))
cell_types = list(net.cell_types.keys())
opt_params = get_conn_params(net.connectivity)
opt_params_bounds = np.tile([[min_weight, max_weight],
                             [min_lamtha, max_lamtha]],
                            (opt_params.shape[0] // 2, 1))

###############################################################################
# simulation
net = L6_model(connect_layer_6=True, legacy_mode=False, grid_shape=(10, 10))
cell_types = list(net.cell_types.keys())
net.add_poisson_drive(name='baseline_drive',
                      rate_constant=poiss_freq,
                      location='soma',
                      n_drive_cells='n_cells',
                      cell_specific=True,
                      weights_ampa={cell: poiss_weight for cell in cell_types},
                      synaptic_delays=0.0,
                      space_constant=1e14,  # diminish impact of space constant
                      probability=1.0,
                      event_seed=poiss_seed)

with MPIBackend(n_procs=10):
    dpls = simulate_dipole(net, tstop=sim_time, n_trials=1)


###############################################################################
# plot results
net_response_fig = plot_net_response(dpls, net)
sr_profiles_fig = plot_sr_profiles(net, sim_time, burn_in_time)

plt.show()
