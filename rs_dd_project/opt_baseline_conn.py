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
from skopt.plots import plot_objective

from hnn_core import pick_connection
from hnn_core.network_models import L6_model
from optimization_lib import (plot_net_response, plot_spiking_profiles,
                              simulate_network, opt_baseline_spike_rates_2,
                              get_conn_params, set_conn_params,
                              plot_convergence)

###############################################################################
# %% set parameters
output_dir = '/users/rthorpe/data/rthorpe/hnn_core_opt_output'

# drive parameters
# 1 kHz as in Billeh et al. 2020 is too fast for this size of network
# decreasing to 10 Hz seems to allow for random single-cell events in a
# disconnected network
poiss_rate = 1e1
# take weights from opt_baseline_drive_refine.py
poiss_weights = dict(L2_basket=6.289657993813411703e-04,
                     L2_pyramidal=9.358499249274205602e-04,
                     L5_basket=9.886578056957024199e-04,
                     L5_pyramidal=2.991545730497045634e-03,
                     L6_basket=9.394567594233604315e-04,
                     L6_pyramidal=9.406783797610669077e-04)

poiss_params = list(poiss_weights.values()) + [poiss_rate]

min_weight, max_weight = 1e-5, 1e-1  # will opt over log_10 domain
#min_lamtha, max_lamtha_i, max_lamtha_e = 1., 5.5, 10.0
lamtha = 6.0

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
sim_time = 2300  # ms
burn_in_time = 300  # ms
rng = np.random.default_rng(1234)
net_original = L6_model(connect_layer_6=True, legacy_mode=False,
                        grid_shape=(12, 12))

# set spatial constant for all local network connections to a single value
lamthas = np.ones_like(net_original.connectivity) * lamtha
set_conn_params(net_original, conn_params=lamthas, weights=False, lamthas=True)

# opt parameters
opt_seq = [
    {'varied': ['L2_basket',
                'L2_pyramidal'],
     'evaluated': ['L2_basket',
                   'L2_pyramidal']},
    {'varied': ['L5_basket',
                'L5_pyramidal'],
     'evaluated': ['L2_basket',
                   'L2_pyramidal',
                   'L5_basket',
                   'L5_pyramidal']},
    {'varied': ['L6_basket', 'L6_pyramidal'],
     'evaluated': ['L2_basket',
                   'L2_pyramidal',
                   'L5_basket',
                   'L5_pyramidal',
                   'L6_basket',
                   'L6_pyramidal']}
]
opt_n_init_points = 300  # < opt_n_total_calls
opt_n_total_calls = 600

all_cell_types = list(net_original.cell_types.keys())
for step_idx, step_cell_types in enumerate(opt_seq):
    ###########################################################################
    # %% get relevant info for current optimization step
    src_types = step_cell_types['varied']
    src_gids = list()
    for cell_type in src_types:
        src_gids.extend(list(net_original.gid_ranges[cell_type]))

    targ_types = step_cell_types['evaluated']
    targ_gids = list()
    for cell_type in targ_types:
        targ_gids.extend(list(net_original.gid_ranges[cell_type]))

    conn_idxs = pick_connection(net_original, src_gids=src_gids,
                                target_gids=targ_gids)

    varied_conns = [net_original.connectivity[conn_idx] for conn_idx in
                    conn_idxs]

    # silence all connections
    # only the varied conns will contribute to this opt step
    net_step = net_original.copy()
    for conn in net_step.connectivity:
        conn['nc_dict']['A_weight'] = 0.0

    ###########################################################################
    # %% set initial parameters and parameter bounds prior
    opt_params_0 = get_conn_params(varied_conns, weights=True, lamthas=False)

    # local network connectivity synaptic weight bounds
    opt_params_bounds = np.tile([min_weight, max_weight],
                                (len(opt_params_0), 1)).tolist()

    ###########################################################################
    # %% prepare cost function
    sim_params = {'sim_time': sim_time, 'burn_in_time': burn_in_time,
                  'n_procs': n_procs, 'poiss_params': poiss_params, 'rng': rng}
    opt_min_func = partial(opt_baseline_spike_rates_2, net=net_step,
                           sim_params=sim_params,
                           target_avg_spike_rates=target_avg_spike_rates)

    ###########################################################################
    # %% optimize
    opt_results = gp_minimize(func=opt_min_func,
                              dimensions=opt_params_bounds,
                              x0=opt_params_0,
                              n_calls=opt_n_total_calls,
                              n_initial_points=opt_n_init_points,
                              initial_point_generator='lhs',  # sobol; params<40
                              acq_func='EI',
                              acq_optimizer='lbfgs',
                              xi=0.01,
                              noise=1e-10,
                              verbose=True,
                              random_state=1)
    opt_params = opt_results.x.copy()
#header = [weight + '_weight' for weight in poiss_weights_ub]
#header = ','.join(header)
#np.savetxt(op.join(output_dir, 'optimized_lamtha_params.csv'),
#           X=[opt_params], delimiter=',', header=header)
np.savetxt(op.join(output_dir, 'optimized_conn_weight_params.csv'),
           X=[opt_params], delimiter=',')
print(f'conn weight params: {opt_params}')

###############################################################################
# %% plot results
ax_converg = plot_convergence(opt_results, ax=None)
fig_converge = ax_converg.get_figure()
plt.tight_layout()
fig_converge.savefig(op.join(output_dir, 'convergence.png'))

ax_objective = plot_objective(opt_results, minimum='expected_minimum')
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

fig_sr_profiles = plot_spiking_profiles(
    net_0, sim_time, burn_in_time, target_spike_rates_1=target_avg_spike_rates,
    target_spike_rates_2=target_avg_spike_rates
)
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

fig_sr_profiles = plot_spiking_profiles(
    net, sim_time, burn_in_time, target_spike_rates_1=target_avg_spike_rates,
    target_spike_rates_2=target_avg_spike_rates
)
plt.tight_layout()
fig_sr_profiles.savefig(op.join(output_dir, 'post_opt_spikerate_profile.png'))

print('local net connectivity (lamtha) optimization routine completed successfully!!!')
