"""Optimize the baseline Poisson drive to the Network.

Designed to be run in a batch script.
"""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>

import os.path as op
from collections import OrderedDict
from functools import partial

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective

from hnn_core.network_models import L6_model
from optimization import (plot_net_response, plot_spiking_profiles,
                          simulate_network, opt_baseline_spike_rates)

###############################################################################
# set parameters
output_dir = '/users/rthorpe/data/rthorpe/hnn_core_opt_output'

# drive parameters
# note that basket cells and pyramidal cells require different amounts of AMPA
# excitatory current in order to drive a spike
poiss_weights_0 = OrderedDict(L2_basket=1e-3, L2_pyramidal=2e-3,
                              L5_basket=1e-3, L5_pyramidal=2e-3,
                              L6_basket=1e-3, L6_pyramidal=2e-3)
# 1 kHz as in Billeh et al. 2020 is too fast for this size of network
# decreasing to 10 Hz seems to allow for random single-cell events in a
# disconnected network
poiss_rate_0 = 1e1
# log_10 scale
min_weight, max_weight = -5., -1.
# real number scale
min_lamtha, max_lamtha = 1., 100.
min_rate, max_rate = 1., 100.

# simulation parameters
n_procs = 24  # parallelize simulation
sim_time = 600  # ms
burn_in_time = 200  # ms
net_original = L6_model(connect_layer_6=True, legacy_mode=False,
                        grid_shape=(10, 10))

# opt parameters
opt_n_total_calls = 128 + 20  # should converge rather quickly...
opt_n_init_points = 128  # 2 ** n_params, 2 samples per dimension in hypercube

###############################################################################
# get initial params prior to optimization
#opt_params_0 = get_conn_params(net_original.connectivity)
# poisson drive synaptic weight initial conditions
opt_params_0 = [np.log10(weight) for weight in poiss_weights_0.values()]
# poisson drive rate constant initial conditions
opt_params_0.append(poiss_rate_0)
# poisson drive synaptic weight bounds
opt_params_bounds = np.tile([min_weight, max_weight],
                            (len(poiss_weights_0), 1)).tolist()
# poisson drive rate constant bounds
opt_params_bounds.append([min_rate, max_rate])

###############################################################################
# prepare cost function
sim_params = {'sim_time': sim_time, 'burn_in_time': burn_in_time,
              'n_procs': n_procs}
opt_min_func = partial(opt_baseline_spike_rates, net=net_original.copy(),
                       sim_params=sim_params)

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
print(f'poiss_weights: {[10 ** param for param in opt_params[:-1]]}')
print(f'poiss_rate: {opt_params[-1]}')

###############################################################################
# plot results

ax_converg = plot_convergence(opt_results, ax=None)
fig_converge = ax_converg.get_figure()
plt.tight_layout()
fig_converge.savefig(op.join(output_dir, 'convergence.png'))

ax_objective = plot_objective(opt_results)
fig_objective = ax_objective.get_figure()
plt.tight_layout()
fig_objective.savefig(op.join(output_dir, 'surrogate_objective_func.png'))

# pre-optimization
# first convert first param back from log_10 scale
opt_params_init = [10 ** weight for weight in opt_params_0[:-1]]
opt_params_init.append(opt_params_0[-1])
net_0, dpls_0 = simulate_network(net_original.copy(), sim_time, burn_in_time,
                                 poiss_params=opt_params_init, clear_conn=True)

fig_net_response = plot_net_response(dpls_0, net_0, sim_time)
plt.tight_layout()
fig_net_response.savefig(op.join(output_dir, 'pre_opt_sim.png'))

fig_sr_profiles = plot_spiking_profiles(net_0, sim_time, burn_in_time)
plt.tight_layout()
fig_sr_profiles.savefig(op.join(output_dir, 'pre_opt_spikerate_profile.png'))

# post-optimization
opt_params_final = [10 ** weight for weight in opt_params[:-1]]
opt_params_final.append(opt_params[-1])
net, dpls = simulate_network(net_original.copy(), sim_time, burn_in_time,
                             poiss_params=opt_params_final, clear_conn=True)

fig_net_response = plot_net_response(dpls, net, sim_time)
plt.tight_layout()
fig_net_response.savefig(op.join(output_dir, 'post_opt_sim.png'))

fig_sr_profiles = plot_spiking_profiles(net, sim_time, burn_in_time)
plt.tight_layout()
fig_sr_profiles.savefig(op.join(output_dir, 'post_opt_spikerate_profile.png'))