"""Manually refine the optimized the baseline Poisson drive parameters."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>

import numpy as np
import matplotlib.pyplot as plt

from hnn_core.network_models import L6_model
from optimization_lib import (plot_net_response, plot_spiking_profiles,
                              simulate_network, err_spike_rates_logdiff)

poiss_rate = 1e1
poiss_params = [6.72e-04,
                9.27e-04,
                9.85e-04,
                29.74e-04,
                9.27e-04,
                9.50e-04,
                poiss_rate]

poiss_params = [6.015690085422602555e-04,
                8.870846881656557596e-04,
                9.849717799957562019e-04,
                3.001426886170132681e-03,
                9.220506366139438487e-04,
                9.448952371442351965e-04,
                poiss_rate]

poiss_params = [6.059706538226362774e-04,
                8.854004692889319876e-04,
                9.871726214402615825e-04,
                2.990327807178780068e-03,
                9.314732985709431876e-04,
                9.440639072517770453e-04,
                poiss_rate]

n_procs = 12
sim_time = 2300
burn_in_time = 300
n_trials = 1
rng = np.random.default_rng()

# taken from Reyes-Puerta 2015 and De Kock 2007
# see Constantinople and Bruno 2013 for laminar difference in E-cell
# excitability and proportion of connected pairs
target_avg_spike_rates = {'L2_basket': 0.8,
                          'L2_pyramidal': 0.3,
                          'L5_basket': 2.4,  # L5A + L5B avg
                          'L5_pyramidal': 1.4,  # L5A + L5B avg
                          'L6_basket': 1.3,  # estimated; Reyes-Puerta 2015
                          'L6_pyramidal': 0.5}  # from De Kock 2007
# avg rates in unconn network should be a bit less
# try 33% of the avg rates in a fully connected network
target_sr_unconn = {cell: rate * 0.33 for cell, rate in
                    target_avg_spike_rates.items()}

net_original = L6_model(connect_layer_6=True, legacy_mode=False,
                        grid_shape=(10, 10))
net, dpls = simulate_network(net_original.copy(), sim_time, burn_in_time,
                             poiss_params=poiss_params, clear_conn=True,
                             n_trials=n_trials, n_procs=n_procs, rng=rng)

fig_net_response = plot_net_response(dpls, net)
plt.tight_layout()

fig_sr_profiles = plot_spiking_profiles(
    net, sim_time, burn_in_time, target_spike_rates_1=target_sr_unconn,
    target_spike_rates_2=target_avg_spike_rates
)
plt.tight_layout()
plt.show()

err = err_spike_rates_logdiff(net, sim_time, burn_in_time, target_sr_unconn)
print(f'spike rate profile error: {err}')  # compare to -17.2157 from optimization minimum