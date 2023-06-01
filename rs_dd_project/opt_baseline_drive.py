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

from scipy import optimize
from skopt import gp_minimize, gbrt_minimize, expected_minimum
from skopt.plots import plot_objective, plot_evaluations

from hnn_core.network_models import L6_model
from optimization_lib import (plot_net_response, plot_spiking_profiles,
                              simulate_network, opt_baseline_spike_rates_1,
                              plot_convergence)

###############################################################################
# %% set parameters
output_dir = '/users/rthorpe/data/rthorpe/hnn_core_opt_output'
#output_dir = '/home/ryan/Desktop/stuff'

# drive parameters
# note that basket cells and pyramidal cells require different amounts of AMPA
# excitatory current in order to drive a spike
poiss_weights_lb = OrderedDict(L2_basket=5e-4, L2_pyramidal=6e-4,
                               L5_basket=5e-4, L5_pyramidal=20e-4,
                               L6_basket=5e-4, L6_pyramidal=6e-4)
poiss_weights_ub = OrderedDict(L2_basket=10e-4, L2_pyramidal=20e-4,
                               L5_basket=10e-4, L5_pyramidal=50e-4,
                               L6_basket=10e-4, L6_pyramidal=20e-4)
# 1 kHz as in Billeh et al. 2020 is too fast for this size of network
# decreasing to 10 Hz seems to allow for random single-cell events in a
# disconnected network
poiss_rate = 1e1

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

# simulation parameters
n_procs = 32  # parallelize simulation
sim_time = 2300  # ms
burn_in_time = 300  # ms
#rng = np.random.default_rng(1234)
rng = 1234  # use a consistently seeded rng for every iteration
net_original = L6_model(connect_layer_6=True, legacy_mode=False,
                        grid_shape=(12, 12))

# opt parameters
opt_n_init_points = 200  # >2 ** n_params, 2 samples per dimension in hypercube
opt_n_total_calls = 600  # >opt_n_init_points

###############################################################################
# %% set initial parameters and parameter bounds prior
opt_params_0 = list(poiss_weights_ub.values())
opt_params_bounds = list(zip(poiss_weights_lb.values(),
                             poiss_weights_ub.values()))

###############################################################################
# %% prepare cost function
sim_params = {'sim_time': sim_time, 'burn_in_time': burn_in_time,
              'n_procs': n_procs, 'poiss_rate_constant': poiss_rate,
              'rng': rng}
opt_min_func = partial(opt_baseline_spike_rates_1,
                       net=net_original,
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
                          n_calls=opt_n_total_calls,
                          n_initial_points=opt_n_init_points,
                          initial_point_generator='lhs',  # sobol; params<40
                          acq_func='EI',
                          acq_optimizer='lbfgs',
                          xi=0.01,
                          noise=1e-10,
                          verbose=True,
                          random_state=1)

# get the last min of the surrogate function, not the min sampled observation
opt_params = opt_results.x.copy()
# get the location and value of the expected minimum of the surrogate function
ev_params, ev_cost = expected_minimum(opt_results, n_random_starts=20,
                                      random_state=1)
#opt_params = ev_params.copy()
header = [weight + '_weight' for weight in poiss_weights_ub]
header = ','.join(header)
np.savetxt(op.join(output_dir, 'optimized_baseline_drive_params.csv'),
           X=[opt_params, ev_params], delimiter=',', header=header)
print(f'poiss_weights: {opt_params}')
#print(f'distance from target: {ev_cost}')

###############################################################################
# %% plot results
ax_eval = plot_evaluations(opt_results)
fig_eval = ax_eval[0, 0].get_figure()
plt.tight_layout()
fig_eval.savefig(op.join(output_dir, 'param_evaluations.png'))

ax_converg = plot_convergence(opt_results, ax=None)
fig_converge = ax_converg.get_figure()
plt.tight_layout()
fig_converge.savefig(op.join(output_dir, 'convergence.png'))

ax_objective = plot_objective(opt_results, n_samples=1000,
                              minimum='expected_minimum')
fig_objective = ax_objective[0, 0].get_figure()
plt.tight_layout()
fig_objective.savefig(op.join(output_dir, 'surrogate_objective_func.png'))

# pre-optimization
poiss_params_init = np.append(opt_params_0, poiss_rate)
rng = np.random.default_rng(1234)
net_0, dpls_0 = simulate_network(net_original.copy(), sim_time, burn_in_time,
                                 n_procs=n_procs,
                                 poiss_params=poiss_params_init,
                                 clear_conn=True,
                                 rng=rng)

fig_net_response = plot_net_response(dpls_0, net_0)
plt.tight_layout()
fig_net_response.savefig(op.join(output_dir, 'pre_opt_sim.png'))

fig_sr_profiles = plot_spiking_profiles(
    net_0, sim_time, burn_in_time, target_spike_rates_1=target_sr_unconn,
    target_spike_rates_2=target_avg_spike_rates
)
plt.tight_layout()
fig_sr_profiles.savefig(op.join(output_dir, 'pre_opt_spikerate_profile.png'))

# post-optimization
poiss_params = np.append(opt_params, poiss_rate)
rng = np.random.default_rng(1234)
net, dpls = simulate_network(net_original.copy(), sim_time, burn_in_time,
                             n_procs=n_procs, poiss_params=poiss_params,
                             clear_conn=True,
                             rng=rng)

fig_net_response = plot_net_response(dpls, net)
plt.tight_layout()
fig_net_response.savefig(op.join(output_dir, 'post_opt_sim.png'))

fig_sr_profiles = plot_spiking_profiles(
    net, sim_time, burn_in_time, target_spike_rates_1=target_sr_unconn,
    target_spike_rates_2=target_avg_spike_rates
)
plt.tight_layout()
fig_sr_profiles.savefig(op.join(output_dir, 'post_opt_spikerate_profile.png'))

print('baseline drive optimization routine completed successfully!!!')
