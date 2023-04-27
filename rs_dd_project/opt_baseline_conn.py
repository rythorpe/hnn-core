"""Optimize Network connectivity to match baseline spiking.

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
from optimization_lib import (plot_net_response, plot_spiking_profiles,
                              simulate_network, opt_baseline_spike_rates_2,
                              get_conn_params)

###############################################################################
# %% set parameters
output_dir = '/users/rthorpe/data/rthorpe/hnn_core_opt_output'

# drive parameters
# 1 kHz as in Billeh et al. 2020 is too fast for this size of network
# decreasing to 10 Hz seems to allow for random single-cell events in a
# disconnected network
poiss_rate = 1e1
# take weights from opt_baseline_drive_refine.py
poiss_weights = dict(L2_basket=6.70e-04, L2_pyramidal=9.23e-04,
                     L5_basket=9.86e-04, L5_pyramidal=29.71e-04,
                     L6_basket=9.15e-04, L6_pyramidal=9.50e-04)
poiss_params = list(poiss_weights.values()) + [poiss_rate]

#min_weight, max_weight = 1e-5, 1e-1  # will opt over log_10 domain
min_lamtha, max_lamtha = 1., 100.

# taken from Reyes-Puerta 2015 and De Kock 2007
# see Constantinople and Bruno 2013 for laminar difference in E-cell
# excitability and proportion of connected pairs
target_avg_spike_rates = {'L2_basket': 0.8,
                          'L2_pyramidal': 0.3,
                          'L5_basket': 2.4,  # L5A + L5B avg
                          'L5_pyramidal': 1.4,  # L5A + L5B avg
                          'L6_basket': 1.3,  # estimated; Reyes-Puerta 2015
                          'L6_pyramidal': 0.5}  # from De Kock 2007

# simulation parameters
n_procs = 32  # parallelize simulation
sim_time = 2200  # ms
burn_in_time = 200  # ms
rng = np.random.default_rng(1234)
net_original = L6_model(connect_layer_6=True, legacy_mode=False,
                        grid_shape=(10, 10))

# opt parameters
opt_n_init_points = 100  # < opt_n_total_calls
opt_n_total_calls = 500

###############################################################################
# %% set initial parameters and parameter bounds prior
opt_params_0 = get_conn_params(net_original.connectivity, weights=False,
                               lamthas=True)

# local network connectivity lamtha bounds
opt_params_bounds = np.tile([min_lamtha, max_lamtha],
                            (len(opt_params_0), 1)).tolist()

###############################################################################
# %% prepare cost function
sim_params = {'sim_time': sim_time, 'burn_in_time': burn_in_time,
              'n_procs': n_procs, 'poiss_params': poiss_params, 'rng': rng}
opt_min_func = partial(opt_baseline_spike_rates_2, net=net_original.copy(),
                       sim_params=sim_params,
                       target_avg_spike_rates=target_avg_spike_rates)

###############################################################################
# %% optimize
opt_results = gp_minimize(func=opt_min_func,
                          dimensions=opt_params_bounds,
                          x0=opt_params_0,
                          n_calls=opt_n_total_calls,
                          n_initial_points=opt_n_init_points,
                          initial_point_generator='lhs',  # sobol; params<40
                          acq_optimizer='sampling',
                          verbose=True,
                          random_state=1234)
opt_params = opt_results.x.copy()
#header = [weight + '_weight' for weight in poiss_weights_ub]
#header = ','.join(header)
#np.savetxt(op.join(output_dir, 'optimized_lamtha_params.csv'),
#           X=[opt_params], delimiter=',', header=header)
np.savetxt(op.join(output_dir, 'optimized_lamtha_params.csv'),
           X=[opt_params], delimiter=',')
print(f'lamtha params: {opt_params}')

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
net_0, dpls_0 = simulate_network(net_original.copy(), sim_time, burn_in_time,
                                 n_procs=n_procs,
                                 poiss_params=poiss_params,
                                 conn_params=opt_params_0,
                                 clear_conn=False,
                                 rng=rng)

fig_net_response = plot_net_response(dpls_0, net_0)
plt.tight_layout()
fig_net_response.savefig(op.join(output_dir, 'pre_opt_sim.png'))

fig_sr_profiles = plot_spiking_profiles(net_0, sim_time, burn_in_time,
                                        target_spike_rates=target_avg_spike_rates)
plt.tight_layout()
fig_sr_profiles.savefig(op.join(output_dir, 'pre_opt_spikerate_profile.png'))

# post-optimization
net, dpls = simulate_network(net_original.copy(), sim_time, burn_in_time,
                             n_procs=n_procs,
                             poiss_params=poiss_params,
                             conn_params=opt_params,
                             clear_conn=False,
                             rng=rng)

fig_net_response = plot_net_response(dpls, net)
plt.tight_layout()
fig_net_response.savefig(op.join(output_dir, 'post_opt_sim.png'))

fig_sr_profiles = plot_spiking_profiles(net, sim_time, burn_in_time,
                                        target_spike_rates=target_avg_spike_rates)
plt.tight_layout()
fig_sr_profiles.savefig(op.join(output_dir, 'post_opt_spikerate_profile.png'))

print('local net connectivity (lamtha) optimization routine completed successfully!!!')
