"""Network model functions."""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>

import os.path as op
import numpy as np
import hnn_core
from hnn_core import read_params
from .network import Network
from .params import _short_name
from .cells_default import pyramidal_ca
from .externals.mne import _validate_type


def jones_2009_model(params=None, add_drives_from_params=False,
                     legacy_mode=False):
    """Instantiate the network model described in
    Jones et al. J. of Neurophys. 2009 [1]_

    Parameters
    ----------
    params : str | dict | None
        The path to the parameter file for constructing the network.
        If None, parameters loaded from default.json
        Default: None
    add_drives_from_params : bool
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False
    legacy_mode : bool
        Set to False by default. Enables matching HNN GUI output when drives
        are added suitably. Will be deprecated in a future release.

    Returns
    -------
    net : Instance of Network object
        Network object used to store

    Notes
    -----
    The network is composed of a square grid of pyramidal cells, arranged in
    two layers (L5 and L2). The default in-plane separation of the grid points
    is 1.0 um, and the layer separation 1307.4 um. These can be adjusted after
    the net is created using the set_cell_positions-method. An all-to-all
    connectivity pattern is applied between cells. Inhibitory basket cells are
    present at a 1:3-ratio.

    References
    ----------
    .. [1] Jones, Stephanie R., et al. "Quantitative Analysis and
           Biophysically Realistic Neural Modeling of the MEG Mu Rhythm:
           Rhythmogenesis and Modulation of Sensory-Evoked Responses."
           Journal of Neurophysiology 102, 3554–3572 (2009).

    """
    hnn_core_root = op.dirname(hnn_core.__file__)
    if params is None:
        params = op.join(hnn_core_root, 'param', 'default.json')
    if isinstance(params, str):
        params = read_params(params)

    net = Network(params, add_drives_from_params=add_drives_from_params,
                  legacy_mode=legacy_mode)

    delay = net.delay

    # source of synapse is always at soma

    # layer2 Pyr -> layer2 Pyr
    # layer5 Pyr -> layer5 Pyr
    lamtha = 3.0
    loc = 'proximal'
    for target_cell in ['L2_pyramidal', 'L5_pyramidal']:
        for receptor in ['nmda', 'ampa']:
            key = f'gbar_{_short_name(target_cell)}_'\
                  f'{_short_name(target_cell)}_{receptor}'
            weight = net._params[key]
            net.add_connection(
                target_cell, target_cell, loc, receptor, weight,
                delay, lamtha, allow_autapses=False)

    # layer2 Basket -> layer2 Pyr
    src_cell = 'L2_basket'
    target_cell = 'L2_pyramidal'
    lamtha = 50.
    loc = 'soma'
    for receptor in ['gabaa', 'gabab']:
        key = f'gbar_L2Basket_L2Pyr_{receptor}'
        weight = net._params[key]
        net.add_connection(
            src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer5 Basket -> layer5 Pyr
    src_cell = 'L5_basket'
    target_cell = 'L5_pyramidal'
    lamtha = 70.
    loc = 'soma'
    for receptor in ['gabaa', 'gabab']:
        key = f'gbar_L5Basket_{_short_name(target_cell)}_{receptor}'
        weight = net._params[key]
        net.add_connection(
            src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer2 Pyr -> layer5 Pyr
    src_cell = 'L2_pyramidal'
    lamtha = 3.
    receptor = 'ampa'
    for loc in ['proximal', 'distal']:
        key = f'gbar_L2Pyr_{_short_name(target_cell)}'
        weight = net._params[key]
        net.add_connection(
            src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer2 Basket -> layer5 Pyr
    src_cell = 'L2_basket'
    lamtha = 50.
    key = f'gbar_L2Basket_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'distal'
    receptor = 'gabaa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # xx -> layer2 Basket
    src_cell = 'L2_pyramidal'
    target_cell = 'L2_basket'
    lamtha = 3.
    key = f'gbar_L2Pyr_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    src_cell = 'L2_basket'
    lamtha = 20.
    key = f'gbar_L2Basket_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'soma'
    receptor = 'gabaa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # xx -> layer5 Basket
    src_cell = 'L5_basket'
    target_cell = 'L5_basket'
    lamtha = 20.
    loc = 'soma'
    receptor = 'gabaa'
    key = f'gbar_L5Basket_{_short_name(target_cell)}'
    weight = net._params[key]
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha,
        allow_autapses=False)

    src_cell = 'L5_pyramidal'
    lamtha = 3.
    key = f'gbar_L5Pyr_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    src_cell = 'L2_pyramidal'
    lamtha = 3.
    key = f'gbar_L2Pyr_{_short_name(target_cell)}'
    weight = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    return net


def law_2021_model(params=None, add_drives_from_params=False,
                   legacy_mode=False):
    """Instantiate the expansion of Jones 2009 model to study beta
    modulated ERPs as described in
    Law et al. Cereb. Cortex 2021 [1]_

    Returns
    -------
    net : Instance of Network object
        Network object used to store the model used in
        Law et al. 2021.

    See Also
    --------
    jones_2009_model

    Notes
    -----
    Model reproduces results from Law et al. 2021
    This model differs from the default network model in several
    parameters including
    1) Increased GABAb time constants on L2/L5 pyramidal cells
    2) Decrease L5_pyramidal -> L5_pyramidal nmda weight
    3) Modified L5_basket -> L5_pyramidal inhibition weights
    4) Removal of L5 pyramidal somatic and basal dendrite calcium channels
    5) Replace L2_basket -> L5_pyramidal GABAa connection with GABAb
    6) Addition of L5_basket -> L5_pyramidal distal connection

    References
    ----------
    .. [1] Law, Robert G., et al. "Thalamocortical Mechanisms Regulating the
           Relationship between Transient Beta Events and Human Tactile
           Perception." Cerebral Cortex, 32, 668–688 (2022).
    """

    net = jones_2009_model(params, add_drives_from_params, legacy_mode)

    # Update biophysics (increase gabab duration of inhibition)
    net.cell_types['L2_pyramidal'].synapses['gabab']['tau1'] = 45.0
    net.cell_types['L2_pyramidal'].synapses['gabab']['tau2'] = 200.0
    net.cell_types['L5_pyramidal'].synapses['gabab']['tau1'] = 45.0
    net.cell_types['L5_pyramidal'].synapses['gabab']['tau2'] = 200.0

    # Decrease L5_pyramidal -> L5_pyramidal nmda weight
    net.connectivity[2]['nc_dict']['A_weight'] = 0.0004

    # Modify L5_basket -> L5_pyramidal inhibition
    net.connectivity[6]['nc_dict']['A_weight'] = 0.02  # gabaa
    net.connectivity[7]['nc_dict']['A_weight'] = 0.005  # gabab

    # Remove L5 pyramidal somatic and basal dendrite calcium channels
    for sec in ['soma', 'basal_1', 'basal_2', 'basal_3']:
        del net.cell_types['L5_pyramidal'].sections[
            sec].mechs['ca']

    # Remove L2_basket -> L5_pyramidal gabaa connection
    del net.connectivity[10]  # Original paper simply sets gbar to 0.0

    # Add L2_basket -> L5_pyramidal gabab connection
    delay = net.delay
    src_cell = 'L2_basket'
    target_cell = 'L5_pyramidal'
    lamtha = 50.
    weight = 0.0002
    loc = 'distal'
    receptor = 'gabab'
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # Add L5_basket -> L5_pyramidal distal connection
    # ("Martinotti-like recurrent tuft connection")
    src_cell = 'L5_basket'
    target_cell = 'L5_pyramidal'
    lamtha = 70.
    loc = 'distal'
    receptor = 'gabaa'
    key = f'gbar_L5Basket_L5Pyr_{receptor}'
    weight = net._params[key]
    net.add_connection(
        src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    return net


# Remove params argument after updating examples
# (only relevant for Jones 2009 model)
def calcium_model(params=None, add_drives_from_params=False,
                  legacy_mode=False):
    """Instantiate the Jones 2009 model with improved calcium dynamics in
    L5 pyramidal neurons. For more details on changes to calcium dynamics
    see Kohl et al. Brain Topragr 2022 [1]_

    Returns
    -------
    net : Instance of Network object
        Network object used to store the Jones 2009 model with an impoved
        calcium channel distribution.

    See Also
    --------
    jones_2009_model

    Notes
    -----
    This model builds on the Jones 2009 model by using a more biologically
    accurate distribution of calcium channels on L5 pyramidal cells.
    Specifically, this model introduces a distance dependent maximum
    conductance (gbar) on calcium channels such that the gbar linearly
    decreases along the dendrites in the direction of the soma.

    References
    ----------
    .. [1] Kohl, Carmen, et al. "Neural Mechanisms Underlying Human Auditory
           Evoked Responses Revealed By Human Neocortical Neurosolver."
           Brain Topography, 35, 19–35 (2022).
    """
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    if params is None:
        params = read_params(params_fname)

    net = jones_2009_model(params, add_drives_from_params, legacy_mode)

    # Replace L5 pyramidal cell template with updated calcium
    cell_name = 'L5_pyramidal'
    pos = net.cell_types[cell_name].pos
    net.cell_types[cell_name] = pyramidal_ca(
        cell_name=_short_name(cell_name), pos=pos)

    return net


def L6_model(params=None, add_drives_from_params=False,
             legacy_mode=False, layer_6_fb=True, grid_shape=(12, 12)):
    """Instantiate the updated calcium model with layer 6 cell types.

    Returns
    -------
    net : Instance of Network object
        Network object used to store the modified Jones 2009 model with an
        impoved calcium channel distribution, layer 6, and other anatomical
        changes needed to interrogate stimulus specific adaptation (SSA) and
        deviance detection (DD).

    See Also
    --------
    jones_2009_model, calcium_model

    Notes
    -----
    This model builds on the updated calcium dynamics model by changing the
    following:
    1) added L6 pyramidal and L6 basket cells plus connections. See Zarrinpar
       and Callaway 2005 and Thomson 2010 anatomical details.
    2) L6 gets activated by proximal drive and propogation of excitation from
       L5
    3) If layer_6_fb is True, a subset of L6 inhibitory interneurons feed
       back onto L5 and L2/3 subpopulations
    """
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    if params is None:
        params = read_params(params_fname)

    # increase size of network: 12x12 instead of 10x10
    params['N_pyr_x'] = grid_shape[0]
    params['N_pyr_y'] = grid_shape[1]

    # XXX HACK: L6 pyramidal and basket cells are added in network.py
    net = Network(params, add_drives_from_params=add_drives_from_params,
                  legacy_mode=legacy_mode)
    # use L5 pyramidal cell as in updated calcium model
    pos = net.cell_types['L5e'].pos
    net.cell_types['L5e'] = pyramidal_ca(cell_name='L5Pyr', pos=pos)

    # Update biophysics (increase gabab duration of inhibition) as in Law model
    # for cell_type in net.cell_types.keys():
    #     if 'pyramidal' in cell_type:
    #         net.cell_types[cell_type].synapses['gabab']['tau1'] = 45.0
    #         net.cell_types[cell_type].synapses['gabab']['tau2'] = 200.0

    conn_weights = {"L2e_L2e_ampa": 0.00066,  # 0.00070
                    "L2e_L2e_nmda": 0.00001,
                    "L2i_L2e_gabaa": 0.0040,
                    "L2i_L2e_gabab": 0.0020,
                    "L2e_L2i_ampa": 0.00108,  # 0.00090
                    "L2i_L2i_gabaa": 0.02,
                    "L6i_cross_L2e_gabaa": 0.05,
                    "L2e_L5e_ampa": 0.0008,
                    "L2i_L5e_gabaa": 0.002,
                    "L5e_L5e_ampa": 0.0009,  # 0.00077
                    "L5e_L5e_nmda": 0.00001,
                    "L5i_L5e_gabaa": 0.01065,  # 0.018
                    "L5i_L5e_gabab": 0.00005,  # changed from jones09
                    "L6i_cross_L5e_gabaa": 0.01,
                    "L2e_L5i_ampa": 0.00010,
                    "L5e_L5i_ampa": 0.00015,  # 0.00043
                    "L5i_L5i_gabaa": 0.02,
                    "L5e_L6e_ampa": 0.00002,
                    "L6e_L6e_ampa": 0.00063,
                    "L6e_L6e_nmda": 0.00001,
                    "L6i_L6e_gabaa": 0.0040,
                    "L6i_L6e_gabab": 0.0020,
                    "L6e_L6i_ampa": 0.00099,
                    "L6i_L6i_gabaa": 0.02}
    lamtha = 4.0
    lamtha_L6_cross = 8.0
    delay = net.delay
    conn_seed = 1  # using the same seed will enforce matching subpop conn!!!

    #######################################################
    # cell type connections that only have one source group
    #######################################################

    # general connection probabilities
    prob_e_e = 0.33
    prob_i_e = 0.67  # 0.66
    prob_i_i = 0.33  # 0.66
    prob_e_i = 0.67  # 0.66
    prob_i_e_cross = 0.67
    prob_e_e_5 = 0.125

    # layer5 Pyr -> layer5 Pyr
    for receptor in ['nmda', 'ampa']:
        net.add_connection(src_gids='L5e',
                           target_gids='L5e',
                           loc='proximal',
                           receptor=receptor,
                           weight=conn_weights[f'L5e_L5e_{receptor}'],
                           delay=delay,
                           lamtha=lamtha,
                           allow_autapses=False,
                           probability=prob_e_e_5,
                           conn_seed=conn_seed)

    # layer5 Basket -> layer5 Pyr
    for receptor in ['gabaa', 'gabab']:
        net.add_connection(src_gids='L5i',
                           target_gids='L5e',
                           loc='soma',
                           receptor=receptor,
                           weight=conn_weights[f'L5i_L5e_{receptor}'],
                           delay=delay,
                           lamtha=lamtha,
                           probability=prob_i_e,
                           conn_seed=conn_seed)

    # xx -> layer5 Basket
    net.add_connection(src_gids='L5i',
                       target_gids='L5i',
                       loc='soma',
                       receptor='gabaa',
                       weight=conn_weights['L5i_L5i_gabaa'],
                       delay=delay,
                       lamtha=lamtha,
                       allow_autapses=False,
                       probability=prob_i_i,
                       conn_seed=conn_seed)

    net.add_connection(src_gids='L5e',
                       target_gids='L5i',
                       loc='soma',
                       receptor='ampa',
                       weight=conn_weights['L5e_L5i_ampa'],
                       delay=delay,
                       lamtha=lamtha,
                       probability=prob_e_i,
                       conn_seed=conn_seed)

    ######################################################################
    # loop over cell type connections that have more than one source group
    ######################################################################
    for src_group in [1, 2]:
        targ_group = src_group

        # general connection probabilities
        prob_e_e = 0.33
        prob_i_e = 0.67  # 0.66
        prob_i_i = 0.33  # 0.66
        prob_e_i = 0.67  # 0.66

        # layer2 Pyr -> layer5 Pyr
        for loc in ['proximal', 'distal']:
            net.add_connection(src_gids=f'L2e_{src_group}',
                               target_gids='L5e',
                               loc=loc,
                               receptor='ampa',
                               weight=conn_weights['L2e_L5e_ampa'],
                               delay=delay,
                               lamtha=lamtha,
                               probability=prob_e_e,
                               conn_seed=conn_seed)

        # layer2 Basket -> layer5 Pyr
        net.add_connection(src_gids=f'L2i_{src_group}',
                           target_gids='L5e',
                           loc='distal',
                           receptor='gabaa',
                           weight=conn_weights['L2i_L5e_gabaa'],
                           delay=delay,
                           lamtha=lamtha,
                           probability=prob_i_e,
                           conn_seed=conn_seed)

        net.add_connection(src_gids=f'L2e_{src_group}',
                           target_gids='L5i',
                           loc='soma',
                           receptor='ampa',
                           weight=conn_weights['L2e_L5i_ampa'],
                           delay=delay,
                           lamtha=lamtha,
                           probability=prob_e_i,
                           conn_seed=conn_seed)

        # layer5 Pyr -> layer6 Pyr
        for loc in ['proximal', 'deep_basal']:
            net.add_connection(src_gids='L5e',
                               target_gids=f'L6e_{targ_group}',
                               loc=loc,
                               receptor='ampa',
                               weight=conn_weights['L5e_L6e_ampa'],
                               delay=delay,
                               lamtha=lamtha,
                               probability=prob_e_e,
                               conn_seed=conn_seed)
        if layer_6_fb:
            # note: cross-laminar inhibtion from L6i only targets L5e and L2e
            # layer6 Bask -> layer5 Pyr
            src_gid_group = f'L6i_{src_group}'
            gid_idxs = [0, 3, 20, 23]
            gid_range = net.gid_ranges[src_gid_group]
            src_gids = [gid for idx, gid in enumerate(gid_range)
                        if idx in gid_idxs]
            net.add_connection(src_gids=src_gids,
                               target_gids='L5e',
                               loc='soma',
                               receptor='gabaa',
                               weight=conn_weights['L6i_cross_L5e_gabaa'],
                               delay=delay,
                               lamtha=lamtha_L6_cross,
                               probability=prob_i_e_cross,
                               conn_seed=conn_seed)

            # layer6 Bask -> layer2 Pyr (within-group only)
            net.add_connection(src_gids=src_gids,
                               target_gids=f'L2e_{targ_group}',
                               loc='soma',
                               receptor='gabaa',
                               weight=conn_weights['L6i_cross_L2e_gabaa'],
                               delay=delay,
                               lamtha=lamtha_L6_cross,
                               probability=prob_i_e_cross,
                               conn_seed=conn_seed)

        ######################################################################
        # loop over cell type connections that have more than one target group
        ######################################################################
        for targ_group in [1, 2]:
            # excitation is greater within groups
            if src_group == targ_group:
                # within-group connection probabilities
                prob_e_e = 0.33
                prob_i_e = 0.67
                prob_i_i = 0.33
                prob_e_i = 0.67
            else:
                # between-group connection probabilities
                prob_e_e = 0.08
                prob_i_e = 0.90
                prob_i_i = 0.125
                prob_e_i = 0.08

            # layer2 Pyr -> layer2 Pyr
            for receptor in ['nmda', 'ampa']:
                net.add_connection(src_gids=f'L2e_{src_group}',
                                   target_gids=f'L2e_{targ_group}',
                                   loc='proximal',
                                   receptor=receptor,
                                   weight=conn_weights[f'L2e_L2e_{receptor}'],
                                   delay=delay,
                                   lamtha=lamtha,
                                   allow_autapses=False,
                                   probability=prob_e_e,
                                   conn_seed=conn_seed)

            # layer2 Basket -> layer2 Pyr
            for receptor in ['gabaa', 'gabab']:
                net.add_connection(src_gids=f'L2i_{src_group}',
                                   target_gids=f'L2e_{targ_group}',
                                   loc='soma',
                                   receptor=receptor,
                                   weight=conn_weights[f'L2i_L2e_{receptor}'],
                                   delay=delay,
                                   lamtha=lamtha,
                                   probability=prob_i_e,
                                   conn_seed=conn_seed)

            # xx -> layer2 Basket
            net.add_connection(src_gids=f'L2e_{src_group}',
                               target_gids=f'L2i_{targ_group}',
                               loc='soma',
                               receptor='ampa',
                               weight=conn_weights['L2e_L2i_ampa'],
                               delay=delay,
                               lamtha=lamtha,
                               probability=prob_e_i,
                               conn_seed=conn_seed)

            net.add_connection(src_gids=f'L2i_{src_group}',
                               target_gids=f'L2i_{targ_group}',
                               loc='soma',
                               receptor='gabaa',
                               weight=conn_weights['L2i_L2i_gabaa'],
                               delay=delay,
                               lamtha=lamtha,
                               probability=prob_i_i,
                               conn_seed=conn_seed)

            # layer6 Pyr -> layer6 Pyr
            for recep in ['ampa', 'nmda']:
                net.add_connection(src_gids=f'L6e_{src_group}',
                                   target_gids=f'L6e_{targ_group}',
                                   loc='deep_basal',
                                   receptor=recep,
                                   weight=conn_weights[f'L6e_L6e_{recep}'],
                                   delay=delay,
                                   lamtha=lamtha,
                                   allow_autapses=False,
                                   probability=prob_e_e,
                                   conn_seed=conn_seed)

            # layer6 Pyr -> layer6 Bask
            net.add_connection(src_gids=f'L6e_{src_group}',
                               target_gids=f'L6i_{targ_group}',
                               loc='soma',
                               receptor='ampa',
                               weight=conn_weights['L6e_L6i_ampa'],
                               delay=delay,
                               lamtha=lamtha,
                               probability=prob_e_i,
                               conn_seed=conn_seed)

            # layer6 Bask -> layer6 Pyr
            for recep in ['gabaa', 'gabab']:
                net.add_connection(src_gids=f'L6i_{src_group}',
                                   target_gids=f'L6e_{targ_group}',
                                   loc='soma',
                                   receptor=recep,
                                   weight=conn_weights[f'L6i_L6e_{recep}'],
                                   delay=delay,
                                   lamtha=lamtha,
                                   probability=prob_i_e,
                                   conn_seed=conn_seed)

            # layer6 Bask -> layer6 Bask
            net.add_connection(src_gids=f'L6i_{src_group}',
                               target_gids=f'L6i_{targ_group}',
                               loc='soma',
                               receptor='gabaa',
                               weight=conn_weights['L6i_L6i_gabaa'],
                               delay=delay,
                               lamtha=lamtha,
                               probability=prob_i_i,
                               conn_seed=conn_seed)

    return net


def add_erp_drives_to_jones_model(net, tstart=0.0):
    """Add drives necessary for an event related potential (ERP)

    Parameters
    ----------
    net : Instance of Network object
        Network object that will be updated with ERP drives.
        Drives are updated in place.
    tstart : float | int
        Start time of sensory input in ms. (Default 0.0 ms)

    Notes
    -----
    The first proximal input arrives at cortex ~20 ms after sensory
    stimulus. The exact delay depends random number generator due to
    random sampling of times from a gaussian.
    """
    _validate_type(net, Network, 'net', 'Network')
    _validate_type(tstart, (float, int), 'tstart', 'float or int')

    # Add distal drive
    weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': 7e-6,
                       'L5_pyramidal': 0.142300}
    weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317,
                       'L5_pyramidal': 0.080074}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                          'L5_pyramidal': 0.1}
    net.add_evoked_drive(
        'evdist1', mu=63.53 + tstart, sigma=3.85, numspikes=1,
        weights_ampa=weights_ampa_d1, weights_nmda=weights_nmda_d1,
        location='distal', synaptic_delays=synaptic_delays_d1, event_seed=274)

    # Add proximal drives
    weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                       'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                            'L5_basket': 1., 'L5_pyramidal': 1.}
    net.add_evoked_drive(
        'evprox1', mu=26.61 + tstart, sigma=2.47, numspikes=1,
        weights_ampa=weights_ampa_p1, weights_nmda=None, location='proximal',
        synaptic_delays=synaptic_delays_prox, event_seed=544)

    weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840,
                       'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}
    net.add_evoked_drive(
        'evprox2', mu=137.12 + tstart, sigma=8.33, numspikes=1,
        weights_ampa=weights_ampa_p2, location='proximal',
        synaptic_delays=synaptic_delays_prox, event_seed=814)
