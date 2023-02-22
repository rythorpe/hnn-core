"""Optimize local network connectivity for realistic resting spikerates."""

# Author: Ryan Thorpe <ryan_thorpe@brown.edu>

import os.path as op
from urllib.request import urlretrieve

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, MPIBackend, JoblibBackend, read_dipole
from hnn_core.network_models import L6_model, calcium_model
from hnn_core.viz import plot_dipole

hnn_core_root = op.join(op.dirname(hnn_core.__file__))
