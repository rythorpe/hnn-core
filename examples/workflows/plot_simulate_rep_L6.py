
# Author: Ryan Thorpe <ryan_thorpe@brown.edu>


import os.path as op
from urllib.request import urlretrieve

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, MPIBackend, JoblibBackend, read_dipole
from hnn_core.network_models import L6_model
from hnn_core.viz import plot_dipole

data_url = ('https://raw.githubusercontent.com/jonescompneurolab/hnn/master/'
            'data/MEG_detection_data/S1_SupraT.txt')
urlretrieve(data_url, 'S1_SupraT.txt')
hnn_core_root = op.join(op.dirname(hnn_core.__file__))
emp_dpl = read_dipole(op.join(hnn_core_root, 'yes_trial_S1_ERP_all_avg.txt'))
#emp_dpl = read_dipole('S1_SupraT.txt')

###############################################################################
# Let us first create our default network and visualize the cells
# inside it.
net = L6_model()
net.plot_cells()
fig = plt.figure(figsize=(6, 6), constrained_layout=True)
for cell_type_idx, cell_type in enumerate(net.cell_types):
    ax = fig.add_subplot(1, len(net.cell_types), cell_type_idx + 1,
                         projection='3d')
    net.cell_types[cell_type].plot_morphology(ax=ax, show=False)
plt.show()

###############################################################################
# Define repetative drive sequence
reps = 4
rep_interval = 250.  # in ms; 8 Hz
rep_duration = 170.

tstop = reps * max(rep_interval, rep_duration)
rep_start_times = np.arange(0, tstop, rep_interval)

###############################################################################
for rep_idx, rep_time in enumerate(rep_start_times):
    # First, we add a distal evoked drive
    weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007,
                       'L5_pyramidal': 0.142300}
    weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                       'L5_pyramidal': 0.080074}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                          'L5_pyramidal': 0.1}
    net.add_evoked_drive(
        f'evdist1_rep{rep_idx}', mu=rep_time + 63.53, sigma=3.85, numspikes=1,
        weights_ampa=weights_ampa_d1, weights_nmda=weights_nmda_d1,
        location='distal', synaptic_delays=synaptic_delays_d1, event_seed=274)

    ###############################################################################
    # Then, we add two proximal drives
    weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                       'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                            'L5_basket': 1., 'L5_pyramidal': 1.}
    # all NMDA weights are zero; pass None explicitly
    net.add_evoked_drive(
        f'evprox1_rep{rep_idx}', mu=rep_time + 26.61, sigma=2.47, numspikes=1,
        weights_ampa=weights_ampa_p1, weights_nmda=None, location='proximal',
        synaptic_delays=synaptic_delays_prox, event_seed=544)

    # Second proximal evoked drive. NB: only AMPA weights differ from first
    weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                       'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
    # all NMDA weights are zero; omit weights_nmda (defaults to None)
    net.add_evoked_drive(
        f'evprox2_rep{rep_idx}', mu=rep_time + 137.12, sigma=8.33, numspikes=1,
        weights_ampa=weights_ampa_p2, location='proximal',
        synaptic_delays=synaptic_delays_prox, event_seed=814)

###############################################################################
# Now let's simulate the dipole

with MPIBackend(n_procs=6):
#with JoblibBackend(n_jobs=1):
    dpls = simulate_dipole(net, tstop=tstop, n_trials=1)

window_len, scaling_factor = 30, 3000
for dpl in dpls:
    dpl.smooth(window_len).scale(scaling_factor)

###############################################################################
# Plot the amplitudes of the simulated aggregate dipole moments over time
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
net.cell_response.plot_spikes_hist(ax=axes[0],
                                   spike_types=['evprox', 'evdist'])
plot_dipole(dpls, ax=axes[1], layer='agg', show=False)
emp_dpl.plot(ax=axes[1], color='purple')
net.cell_response.plot_spikes_raster(ax=axes[2], show=False)



###############################################################################
# If you want to analyze how the different cortical layers contribute to
# different net waveform features, then instead of passing ``'agg'`` to
# ``layer``, you can provide a list of layers to be visualized and optionally
# a list of axes to ``ax`` to visualize the dipole moments separately.
plot_dipole(dpls, average=False, layer=['L2', 'L5', 'L6', 'agg'], show=False)
plt.show()
