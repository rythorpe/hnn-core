"""Optimize local network connectivity for realistic resting spikerates."""

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

hnn_core_root = op.join(op.dirname(hnn_core.__file__))


def firing_rate(spike_train, t_win):
    return len(spike_train) / t_win


def get_conn_params(loc_net_connections):
    conn_params = list()
    for conn in loc_net_connections:
        conn_params.append(conn['nc_dict']['A_weight'])
        conn_params.append(conn['nc_dict']['lamtha'])
    return conn_params


net = L6_model(connect_layer_6=True, legacy_mode=False, grid_shape=(10, 10))

poisson_freq = 1e3  # 1 kHz as in Billeh et al. 2020
all_weight = 8e-5
poisson_weights = {'L2_basket': all_weight, 'L2_pyramidal': all_weight,
                   'L5_basket': all_weight, 'L5_pyramidal': all_weight,
                   'L6_basket': all_weight, 'L6_pyramidal': all_weight}
poisson_seed = 1
net.add_poisson_drive(name='baseline_drive',
                      rate_constant=poisson_freq,
                      location='soma',
                      n_drive_cells='n_cells',
                      cell_specific=True,
                      weights_ampa=poisson_weights,
                      synaptic_delays=0.0,
                      space_constant=1e14,  # diminish impact of space constant
                      probability=1.0,
                      event_seed=poisson_seed)

with MPIBackend(n_procs=6):
    dpls = simulate_dipole(net, tstop=500, n_trials=1)

window_len, scaling_factor = 30, 2000
for dpl in dpls:
    dpl.smooth(window_len).scale(scaling_factor)

###############################################################################
# Plot the amplitudes of the simulated aggregate dipole moments over time
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
net.cell_response.plot_spikes_hist(ax=axes[0], n_bins=200, show=False)

plot_dipole(dpls, ax=axes[1], layer='agg', show=False)
net.cell_response.plot_spikes_raster(ax=axes[2], show=False)


plot_dipole(dpls, average=False, layer=['L2', 'L5', 'L6', 'agg'], show=False)
plt.show()
