"""Optimize local network connectivity for realistic resting spikerates."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
# import antropy as ant
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

from hnn_core import simulate_dipole, MPIBackend
from hnn_core.viz import plot_dipole

# poisson drive parameters
poiss_drive_params = [5.82e-04,
                      8.80e-04,
                      9.61e-04,
                      29.01e-04,
                      6.85e-04,
                      9.30e-04,
                      1e1]

# order matters!!!
cell_groups = {'L2/3i': ['L2i_1', 'L2i_2'],
               'L2/3e': ['L2e_1', 'L2e_2'],
               'L5i': ['L5i'],
               'L5e': ['L5e'],
               'L6i': ['L6i_1', 'L6i_2'],
               'L6e': ['L6e_1', 'L6e_2']}
special_groups = {'L6i_cross': ['L6i_cross1', 'L6i_cross2']}


def get_conn_params(net, param_type):
    """Get list of lamtha or weight parameters for all network connections."""
    conn_params = list()
    for conn in net.connectivity:
        if param_type == 'weight':
            conn_params.append(conn['nc_dict']['A_weight'])
        elif param_type == 'lamtha':
            conn_params.append(conn['nc_dict']['lamtha'])
        else:
            raise ValueError('unknown connection parameter type!!')
    return conn_params


def set_conn_lamthas(net, lamthas, which_conn_idxs):
    """Set spatial decay contant of selected network connections in-place."""
    set_conn_count = 0
    for conn_idx, conn in enumerate(net.connectivity):
        if conn_idx in which_conn_idxs:
            conn['nc_dict']['lamtha'] = lamthas[set_conn_count]
            set_conn_count += 1


def scale_conn_weights(net, scaling_factors, which_conn_idxs):
    """Scale synaptic weights of selected network connections in-place."""
    set_conn_count = 0
    for conn_idx, conn in enumerate(net.connectivity):
        if conn_idx in which_conn_idxs:
            conn['nc_dict']['A_weight'] *= scaling_factors[set_conn_count]
            set_conn_count += 1


def plot_net_response(dpls, net):
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(6, 12))

    # window_len, scaling_factor = 30, 2000
    # for dpl in dpls:
    #     dpl.smooth(window_len).scale(scaling_factor)

    net.cell_response.plot_spikes_hist(ax=axes[0], bin_width=1.0, show=False)
    plot_dipole(dpls, ax=axes[1:5], layer=['L2', 'L5', 'L6', 'agg'],
                show=False)
    # create dictionary of all cell groupings, including cross-laminar L6 types
    cell_types = cell_groups.copy()
    cell_types.update(special_groups)
    # plot raster for all cell types
    net.cell_response.plot_spikes_raster(ax=axes[5], cell_types=cell_types,
                                         show=False)
    return fig


def plot_spiking_profiles(net, sim_time, burn_in_time, target_spike_rates_1,
                          target_spike_rates_2):
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="ticks", rc=custom_params)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # collapse across trials
    n_trials = len(net.cell_response.spike_gids)
    spike_gids = np.concatenate(net.cell_response.spike_gids).flatten()
    spike_times = np.concatenate(net.cell_response.spike_times).flatten()

    pop_layers = list()
    pop_cell_types = list()
    pop_spike_rates = list()
    pop_targets_1 = list()
    pop_targets_2 = list()
    for layer, cell_types in cell_groups.items():
        layer = layer[:-1]  # crop off e/i from end of cell_type
        for cell_type in cell_types:
            if 'i' in cell_type:
                cell_type_ei = 'i'
            else:
                cell_type_ei = 'e'
            cell_type_key = f'{layer}{cell_type_ei}'

            # count spikes for all cells of this type (e or i) for this layer
            for gid in net.gid_ranges[cell_type]:
                gids_after_burn_in = np.array(spike_gids)[spike_times >
                                                          burn_in_time]
                n_spikes = np.sum(gids_after_burn_in == gid)
                # append to list to construct dataframe later
                pop_layers.append(layer)
                pop_cell_types.append(cell_type_ei)
                pop_spike_rates.append((n_spikes / n_trials /
                                        ((sim_time - burn_in_time) * 1e-3)))
                pop_targets_1.append(target_spike_rates_1[cell_type_key])
                pop_targets_2.append(target_spike_rates_2[cell_type_key])

    spiking_df = pd.DataFrame({'layer': pop_layers,
                               'cell type': pop_cell_types,
                               'spike rate': pop_spike_rates,
                               'target rate 1': pop_targets_1,
                               'target rate 2': pop_targets_2})

    ax = sns.barplot(data=spiking_df, x='spike rate', y='layer',
                     hue='cell type', estimator='mean', palette='Greys',
                     errorbar='se', ax=ax)
    # note: eyeball dodge value to match barplot
    # also, setting legend='_nolegend_' doesn't work when hue is set
    ax = sns.pointplot(data=spiking_df, x='target rate 1', y='layer',
                       hue='cell type', join=False, dodge=0.4,
                       palette=['darkred'], markers='|', ax=ax)
    ax = sns.pointplot(data=spiking_df, x='target rate 2', y='layer',
                       hue='cell type', join=False, dodge=0.4,
                       palette=['k'], markers='D', ax=ax)

    ax.set_ylabel('layer')
    ax.set_xlabel('mean single-unit spikes/s')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[4:], labels=labels[4:])

    # make sure calcuation above is consistent with mean_rates method
    # avg_spike_rates = net.cell_response.mean_rates(tstart=burn_in_time,
    #                                                tstop=sim_time,
    #                                                gid_ranges=net.gid_ranges)
    # print(avg_spike_rates)
    # print(spiking_df.groupby(['layer', 'cell type'])['spike rate'].mean())

    return fig


def plot_spiking_profiles_old(net, sim_time, burn_in_time, target_spike_rates,
                              bin_width=None):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(3, 6))
    layers = ['L2/3', 'L5', 'L6']

    for cell_type_idx, cell_type in enumerate(net.cell_types):
        if 'basket' in cell_type:
            color = sns.color_palette('bright')[8]
        else:
            color = sns.color_palette('bright')[7]

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
        layer_idx = cell_type_idx // 2
        if bin_width is None:
            sns.kdeplot(data=spike_rates, ax=axes[layer_idx], color=color,
                        fill=True)
        elif isinstance(bin_width, float):
            sns.kdeplot(data=spike_rates, bw_method=bin_width, color=color,
                        ax=axes[layer_idx], fill=True)
        else:
            raise ValueError('expected bin_width to be of type float, got '
                             f'{type(bin_width)}.')

        axes[layer_idx].axvline(target_spike_rates[cell_type], linestyle=':',
                                color=color)
        axes[layer_idx].set_ylim([0, 1])
        axes[layer_idx].set_xlim([0, 10])
        axes[layer_idx].set_yticks([0, 1])
        axes[layer_idx].set_ylabel('')
        axes[layer_idx].set_title(layers[layer_idx])
    axes[0].set_ylabel('probability')
    axes[-1].set_xlabel('spike rate (Hz)')
    #fig.suptitle('cell population\nspike rates')
    return fig


def plot_spikerate_hist(net, sim_time, burn_in_time, ax):
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="ticks", rc=custom_params)

    layer_by_cell_type = {'L2i': 'L2/3',
                          'L2e': 'L2/3',
                          'L5i': 'L5',
                          'L5e': 'L5',
                          'L6i': 'L6',
                          'L6e': 'L6'}

    pop_layer = list()
    pop_cell_type = list()
    pop_spike_rates = list()
    for cell_type in net.cell_types:
        if 'i' in cell_type:
            cell_type_ei = 'I'
        else:
            cell_type_ei = 'E'
        spike_gids = np.array(net.cell_response.spike_gids[0])  # only 1 trial
        spike_times = np.array(net.cell_response.spike_times[0])  # same

        for gid_idx, gid in enumerate(net.gid_ranges[cell_type]):
            gids_after_burn_in = np.array(spike_gids)[spike_times >
                                                      burn_in_time]
            n_spikes = np.sum(gids_after_burn_in == gid)
            pop_layer.append(layer_by_cell_type[cell_type])
            pop_cell_type.append(cell_type_ei)
            pop_spike_rates.append((n_spikes /
                                    ((sim_time - burn_in_time) * 1e-3)))

    spiking_df = pd.DataFrame({'layer': pop_layer, 'cell type': pop_cell_type,
                               'spike rate': pop_spike_rates})
    ax = sns.barplot(data=spiking_df, x='spike rate', y='layer',
                     hue='cell type', palette='Greys', errorbar='se', ax=ax)

    ax.set_ylabel('layer')
    ax.set_xlabel('mean single-unit spike rate (Hz)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[2:], labels=labels[2:])

    # make sure calcuation above is consistent with mean_rates method
    # avg_spike_rates = net.cell_response.mean_rates(tstart=burn_in_time,
    #                                                tstop=sim_time,
    #                                                gid_ranges=net.gid_ranges)
    # print(avg_spike_rates)
    # print(spiking_df.groupby(['layer', 'cell type'])['spike rate'].mean())

    return ax.get_figure()


def simulate_network(net, sim_time, burn_in_time, n_trials=1, n_procs=6,
                     poiss_params=None, clear_conn=False, rng=None):
    """Add poisson drive to empty network and run simulation."""

    # induce variation between simulations (aside from parameter exploration)
    if rng is None:
        # define new generator with default seed
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        # define new generator with constant seed (e.g., use w/gbrt_minimize)
        rng = np.random.default_rng(rng)
    seed = rng.integers(0, np.iinfo(np.int32).max)

    # when optimizing cell excitability under poisson drive, it's nice to use
    # a disconnected network
    if clear_conn is True:
        print("simulating disconnected network")
        net.clear_connectivity()
    else:
        print("simulating fully-connected network")

    if poiss_params is not None:
        # cell_types = ['L2_basket', 'L2_pyramidal',
        #               'L5_basket', 'L5_pyramidal',
        #               'L6_basket', 'L6_pyramidal']
        # place cell types from all groups in an a list (in order!!)
        poiss_weights = {cell_group: weight for cell_type, weight in
                         zip(cell_groups.values(), poiss_params[:-1])
                         for cell_group in cell_type}
        poiss_rate = poiss_params[-1]
        # add poisson drive with near-uniform spatial spread
        net.add_poisson_drive(name='poisson_drive',
                              rate_constant=poiss_rate,
                              location='soma',
                              n_drive_cells='n_cells',
                              cell_specific=True,
                              weights_ampa=poiss_weights,
                              synaptic_delays=0.0,
                              space_constant=1e50,
                              probability=1.0,
                              event_seed=seed)

    with MPIBackend(n_procs=n_procs):
        dpls = simulate_dipole(net, tstop=sim_time, n_trials=n_trials,
                               baseline_win=[burn_in_time, sim_time])

    return net, dpls


def err_spike_rates_logdiff(net, sim_time, burn_in_time,
                            target_avg_spike_rates):
    """Cost function for matching simulated vs expected avg spike rates.

    Used for optimizing cell excitability under poisson drive.
    """
    avg_spike_rates = net.cell_response.mean_rates(tstart=burn_in_time,
                                                   tstop=sim_time,
                                                   gid_ranges=net.gid_ranges)
    avg_spike_rates_grouped = dict()
    for cell_label, cell_types in cell_groups.items():
        avg_rates = [avg_spike_rates[cell_type] for cell_type in cell_types]
        avg_spike_rates_grouped[cell_label] = np.mean(avg_rates)

    spike_rate_diffs = list()
    for cell_label in cell_groups.keys():
        # convert to log_10 scale to amplify distances close to zero
        # add distance to a number close to 0 to avoid instability as diff -> 0
        log_diff = np.log10(1e-6 + (target_avg_spike_rates[cell_label] -
                            avg_spike_rates_grouped[cell_label]) ** 2)
        spike_rate_diffs.append(log_diff)
        # spike_rate_diffs.append(target_avg_spike_rates[cell_label] -
        #                         avg_spike_rates_grouped[cell_label])

    # return np.linalg.norm(spike_rate_diffs)
    min_log_diff = -6 * len(target_avg_spike_rates)  # theoretic minimum
    # offset so theoretic min is at zero
    return sum(spike_rate_diffs) - min_log_diff


def err_spike_rates_conn(net, sim_time, burn_in_time,
                         target_avg_spike_rates):
    """Cost function for matching simulated vs expected avg spike rates.

    Used for optimizing local network connectivity given poisson drive.
    """
    avg_spike_rates = net.cell_response.mean_rates(tstart=burn_in_time,
                                                   tstop=sim_time,
                                                   gid_ranges=net.gid_ranges)

    avg_spike_rates_grouped = dict()
    for cell_label, cell_types in cell_groups.items():
        avg_rates = [avg_spike_rates[cell_type] for cell_type in cell_types]
        avg_spike_rates_grouped[cell_label] = np.mean(avg_rates)

    # main term: distance between observed and desired mean spike rates
    # across cell types
    spike_rate_diffs = list()
    max_spike_rate_diff = 0.1
    for cell_type in cell_groups.keys():
        diff = (target_avg_spike_rates[cell_type] -
                avg_spike_rates_grouped[cell_type])
        spike_rate_diffs.append(diff)
    spike_rate_diff_norm = (np.linalg.norm(spike_rate_diffs) /
                            (np.sqrt(len(spike_rate_diffs) *
                             max_spike_rate_diff**2)))

    # regularization term: penalize for a spiking renewal process that is too
    # predictable (i.e., rhythmic)

    # concatenate across trials
    spike_times = np.concatenate(net.cell_response.spike_times)
    spike_gids = np.concatenate(net.cell_response.spike_gids)
    # sort by spike times
    sort_idxs = np.argsort(spike_times)
    spike_times = spike_times[sort_idxs]
    spike_gids = spike_gids[sort_idxs]
    # compile inter-event-intervals for each cell that spikes
    iei_agg = list()
    for cell_gid in np.unique(spike_gids):
        iei = np.diff(spike_times[spike_gids == cell_gid]).tolist()
        iei_agg.extend(iei)
    # square of the coefficient of variation (~Fano Factor)
    cv2 = np.var(iei_agg) / np.mean(iei_agg) ** 2
    regularizer = 1 / cv2 - 1  # invert s.t. 0 is Poisson-like, 1 is rhythmic
    # normalize by a user-defined tolerance s.t. this term is weighted
    # evenly with the desired spike rate distance tolerance
    regularizer /= 0.25

    # regularization term: penalize for over or under autocorrelated spike
    # rates over time (via Detrended Fluxuation Analysis)
    # target_hurst = 0.5
    # hurst_tol = 0.05
    # t_win = 5.0  # ms
    # tstart = burn_in_time
    # tstop = sim_time
    # bins = np.arange()
    # spike_rate_ts = np.histogram(net.cell_response.spike_times, bins=np.arange())
    # hurst = ant.detrended_fluctuation(spike_rate_ts)
    # regularizer = np.abs(hurst - target_hurst) / hurst_tol

    return spike_rate_diff_norm + regularizer


def opt_baseline_spike_rates_1(opt_params, net, sim_params,
                               target_avg_spike_rates):
    """Function to minimize during optimization: err in baseline spikerates.

    Stage 1: optimize over Poisson drive parameters
    """
    sim_time = sim_params['sim_time']
    burn_in_time = sim_params['burn_in_time']
    n_procs = sim_params['n_procs']
    poiss_rate = sim_params['poiss_rate_constant']
    rng = sim_params['rng']

    # convert weight param back from log_10 scale
    #poiss_params = np.append(10 ** np.array(opt_params), poiss_rate)
    poiss_params = np.append(opt_params, poiss_rate)
    net_disconn, _ = simulate_network(net.copy(),
                                      sim_time=sim_time,
                                      burn_in_time=burn_in_time,
                                      n_procs=n_procs,
                                      poiss_params=poiss_params,
                                      clear_conn=True,
                                      rng=rng)

    err = err_spike_rates_logdiff(net_disconn, sim_time, burn_in_time,
                                  target_avg_spike_rates)
    return err


def opt_baseline_spike_rates_2(opt_params, net, sim_params,
                               target_avg_spike_rates):
    """Function to minimize during optimization: err in baseline spikerates.

    Stage 2: optimize over local network connectivity parameters
    """
    sim_time = sim_params['sim_time']
    burn_in_time = sim_params['burn_in_time']
    n_procs = sim_params['n_procs']
    poiss_params = sim_params['poiss_params']
    rng = sim_params['rng']
    which_conn_idxs = sim_params['which_conn_idxs']

    net_scaled = net.copy()
    scaling_fctrs = [10.0 ** exp for exp in opt_params]
    scale_conn_weights(net_scaled, scaling_factors=scaling_fctrs,
                       which_conn_idxs=which_conn_idxs)
    net_connected, _ = simulate_network(net_scaled.copy(),
                                        sim_time=sim_time,
                                        burn_in_time=burn_in_time,
                                        n_procs=n_procs,
                                        poiss_params=poiss_params,
                                        clear_conn=False,
                                        rng=rng)

    # note: pass in opt_params (weight scaling factors) to
    # penalize cost function for large connection weights
    err = err_spike_rates_conn(net_connected,
                               sim_time, burn_in_time,
                               target_avg_spike_rates)
    return err


def plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces (modified from skopt).

    Parameters
    ----------
    args[i] :  `OptimizeResult`, list of `OptimizeResult`, or tuple
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    ax : `Axes`, optional
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    true_minimum : float, optional
        The true minimum value of the function, if known.

    yscale : None or string, optional
        The scale for the y-axis.

    Returns
    -------
    ax : `Axes`
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            ax.plot(range(1, n_calls + 1), results.func_vals, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            mins = [[np.min(r.func_vals[:i]) for i in iterations]
                    for r in results]

            for m in mins:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(mins, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if true_minimum:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum or name:
        ax.legend(loc="best")

    return ax
