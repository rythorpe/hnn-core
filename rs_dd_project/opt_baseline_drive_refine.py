"""Manually refine the optimized the baseline Poisson drive parameters."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>

import numpy as np
import matplotlib.pyplot as plt

from hnn_core.network_models import L6_model
from optimization_lib import (plot_net_response, plot_spiking_profiles,
                              simulate_network, err_spike_rates_logdiff)

poiss_rate = 1e1

# manually-tuned results for 33% target rate [old values]
poiss_params = [6.72e-04,
                9.27e-04,
                9.85e-04,
                29.74e-04,
                9.27e-04,
                9.50e-04,
                poiss_rate]

# final optimization results for 33% target rate
poiss_params = [6.289657993813411703e-04,
                9.358499249274205602e-04,
                9.886578056957024199e-04,
                2.991545730497045634e-03,
                9.394567594233604315e-04,
                9.406783797610669077e-04,
                poiss_rate]

# final optimization results for 20% target rate + manual tuning [use this one]
poiss_params = [5.823801023405118966e-04,
                8.617101631605665613e-04,
                9.603200828904341754e-04,
                2.900971997811154900e-03,
                7.081385474760127415e-04,
                9.301705407062075449e-04,
                poiss_rate]

n_procs = 12
sim_time = 2300
burn_in_time = 300
n_trials = 1
rng = np.random.default_rng(1234)

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
# try 20% of the avg rates in a fully connected network
target_sr_unconn = {cell: rate * 0.2 for cell, rate in
                    target_avg_spike_rates.items()}

net = L6_model(connect_layer_6=True, legacy_mode=False, grid_shape=(12, 12))
for conn in net.connectivity:
    conn['nc_dict']['lamtha'] = 1.5
net, dpls = simulate_network(net.copy(), sim_time, burn_in_time,
                             poiss_params=poiss_params, clear_conn=False,
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