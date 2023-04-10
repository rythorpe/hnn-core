"""Optimize the baseline Poisson drive to the Network.

Designed to be run in a batch script.
"""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>

# %% import modules
import os.path as op
from collections import OrderedDict
from functools import partial

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from skopt import gp_minimize, gbrt_minimize
from skopt.plots import plot_convergence, plot_objective

from hnn_core.network_models import L6_model
from optimization import (plot_net_response, plot_spiking_profiles,
                          simulate_network, opt_baseline_spike_rates)

###############################################################################
# %% set parameters
output_dir = '/users/rthorpe/data/rthorpe/hnn_core_opt_output'

# drive parameters
# note that basket cells and pyramidal cells require different amounts of AMPA
# excitatory current in order to drive a spike
poiss_weights_ub = OrderedDict(L2_basket=1e-3, L2_pyramidal=4e-3,
                               L5_basket=1e-3, L5_pyramidal=4e-3,
                               L6_basket=1e-3, L6_pyramidal=4e-3)

poiss_weights_lb = OrderedDict(L2_basket=5.0e-4, L2_pyramidal=8.8e-4,
                               L5_basket=5.0e-4, L5_pyramidal=25.5e-4,
                               L6_basket=5.0e-4, L6_pyramidal=8.8e-4)
# 1 kHz as in Billeh et al. 2020 is too fast for this size of network
# decreasing to 10 Hz seems to allow for random single-cell events in a
# disconnected network
poiss_rate = 1e1
min_weight, max_weight = 1e-5, 1e-1  # will opt over log_10 domain
min_lamtha, max_lamtha = 1., 100.
min_rate, max_rate = 1., 100.

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

# simulation parameters
n_procs = 32  # parallelize simulation
sim_time = 1000  # ms
burn_in_time = 200  # ms
net_original = L6_model(connect_layer_6=True, legacy_mode=False,
                        grid_shape=(10, 10))

# opt parameters
opt_n_init_points = 100  # >2 ** n_params, 2 samples per dimension in hypercube
opt_n_total_calls = 300  # >opt_n_init_points

###############################################################################
# %% set initial parameters and parameter bounds prior
#opt_params_0 = get_conn_params(net_original.connectivity)
# poisson drive synaptic weight initial conditions; log_10 scale
opt_params_0 = [np.log10(weight) for weight in poiss_weights_ub.values()]
# poisson drive synaptic weight bounds
#opt_params_bounds = np.tile([min_weight, max_weight],
#                            (len(poiss_weights_0), 1)).tolist()
# log_10 scale
opt_params_bounds = [(np.log10(lb), np.log10(ub)) for lb, ub in
                     zip(poiss_weights_ub.values(), poiss_weights_ub.values())]

###############################################################################
# %% prepare cost function
sim_params = {'sim_time': sim_time, 'burn_in_time': burn_in_time,
              'n_procs': n_procs, 'poiss_rate_constant': poiss_rate}
opt_min_func = partial(opt_baseline_spike_rates, net=net_original.copy(),
                       sim_params=sim_params,
                       target_avg_spike_rates=target_sr_unconn)

###############################################################################
# %% optimize

#opt_results = gbrt_minimize(func=opt_min_func,
#                            dimensions=opt_params_bounds,
#                            x0=None,  # opt_params_0
#                            n_calls=opt_n_total_calls,
#                            n_initial_points=opt_n_init_points,
#                            initial_point_generator='lhs',  # sobol, params<40
#                            verbose=True,
#                            random_state=1234)
opt_results = gp_minimize(func=opt_min_func,
                          dimensions=opt_params_bounds,
                          x0=None,  # opt_params_0
                          n_calls=opt_n_total_calls,  # >5**n_params
                          n_initial_points=opt_n_init_points,  # 5**n_params
                          initial_point_generator='lhs',  # sobol; params<40
                          acq_func='EI',
                          xi=0.01,
                          acq_optimizer='lbfgs',
                          verbose=True,
                          random_state=1234)
opt_params = opt_results.x
# convert param back from log_10 scale
opt_params = [10 ** weight for weight in opt_params]
header = [weight + '_weight' for weight in poiss_weights_ub]
header = ','.join(header)
np.savetxt(op.join(output_dir, 'optimized_baseline_drive_params.csv'),
           X=[opt_params], delimiter=',', header=header)
print(f'poiss_weights: {opt_params}')

###############################################################################
# %% plot results
ax_converg = plot_convergence(opt_results, ax=None)
fig_converge = ax_converg.get_figure()
plt.tight_layout()
fig_converge.savefig(op.join(output_dir, 'convergence.png'))

ax_objective = plot_objective(opt_results)
fig_objective = ax_objective[0, 0].get_figure()
plt.tight_layout()
fig_objective.savefig(op.join(output_dir, 'surrogate_objective_func.png'))

# pre-optimization
# first convert weight params back from log_10 scale
poiss_params_init = [10 ** weight for weight in opt_params_0] + [poiss_rate]
net_0, dpls_0 = simulate_network(net_original.copy(), sim_time, burn_in_time,
                                 n_procs=n_procs,
                                 poiss_params=poiss_params_init,
                                 clear_conn=True)

fig_net_response = plot_net_response(dpls_0, net_0, sim_time)
plt.tight_layout()
fig_net_response.savefig(op.join(output_dir, 'pre_opt_sim.png'))

fig_sr_profiles = plot_spiking_profiles(net_0, sim_time, burn_in_time,
                                        target_spike_rates=target_sr_unconn)
plt.tight_layout()
fig_sr_profiles.savefig(op.join(output_dir, 'pre_opt_spikerate_profile.png'))

# post-optimization
poiss_params = opt_params + [poiss_rate]
net, dpls = simulate_network(net_original.copy(), sim_time, burn_in_time,
                             n_procs=n_procs, poiss_params=poiss_params,
                             clear_conn=True)

fig_net_response = plot_net_response(dpls, net, sim_time)
plt.tight_layout()
fig_net_response.savefig(op.join(output_dir, 'post_opt_sim.png'))

fig_sr_profiles = plot_spiking_profiles(net, sim_time, burn_in_time,
                                        target_spike_rates=target_sr_unconn)
plt.tight_layout()
fig_sr_profiles.savefig(op.join(output_dir, 'post_opt_spikerate_profile.png'))

print('baseline drive optimization routine completed successfully!!!')
