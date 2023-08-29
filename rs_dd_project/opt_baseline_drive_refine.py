"""Manually refine the optimized the baseline Poisson drive parameters."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>

import numpy as np
import matplotlib.pyplot as plt

from hnn_core.network_models import L6_model
from optimization_lib import (plot_net_response, plot_spiking_profiles,
                              simulate_network, err_spike_rates_logdiff)

poiss_rate = 1e1

# final optimization results for 20% target rate + manual tuning [use this one]
poiss_params = [5.82e-04,
                8.80e-04,
                9.61e-04,
                29.01e-04,
                6.85e-04,
                9.30e-04,
                poiss_rate]

n_procs = 10
sim_time = 2300
burn_in_time = 300
n_trials = 1
clear_conn = False
connect_layer_6 = True
rng = np.random.default_rng(1234)

# avg population spike rates for each layer
# taken from Reyes-Puerta 2015 and De Kock 2007
# see Constantinople and Bruno 2013 for laminar difference in E-cell
# excitability and proportion of connected pairs
target_sr = {'L2/3i': 0.8,
             'L2/3e': 0.3,
             'L5i': 2.4,  # L5A + L5B avg
             'L5e': 1.4,  # L5A + L5B avg
             'L6i': 1.3,  # estimated; Reyes-Puerta 2015
             'L6e': 0.5}  # from De Kock 2007

# avg rates in unconn network should be a bit less
# try 20% of the avg rates in a fully connected network
target_sr_unconn = {cell: rate * 0.2 for cell, rate in
                    target_sr.items()}

net = L6_model(grid_shape=(12, 12), connect_layer_6=connect_layer_6)
net, dpls = simulate_network(net.copy(), sim_time, burn_in_time,
                             poiss_params=poiss_params, clear_conn=clear_conn,
                             n_trials=n_trials, n_procs=n_procs, rng=rng)

fig_net_response = plot_net_response(dpls, net)
plt.tight_layout()

fig_sr_profiles = plot_spiking_profiles(
    net, sim_time, burn_in_time, target_spike_rates_1=target_sr_unconn,
    target_spike_rates_2=target_sr
)
plt.tight_layout()
plt.show()

err = err_spike_rates_logdiff(net, sim_time, burn_in_time, target_sr_unconn)
print(f'spike rate profile error: {err}')  # compare to -17.2157 from optimization minimum
