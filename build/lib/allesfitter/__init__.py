#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:18:20 2018

@author:
Maximilian N. Günther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import os

#:::: allesfitter modules
from .mcmc import mcmc_fit
from .nested_sampling import ns_fit

from .general_output import get_labels
from .nested_sampling_output import get_ns_posterior_samples, ns_output
from .mcmc_output import get_mcmc_posterior_samples, mcmc_output

from .priors import transform_priors
from .priors.estimate_noise import estimate_noise

from .postprocessing.nested_sampling_compare_logZ import ns_plot_bayes_factors
from .postprocessing.plot_violins import ns_plot_violins, mcmc_plot_violins

def GUI():
    allesfitter_path = os.path.dirname( os.path.realpath(__file__) )
    os.system( 'jupyter notebook "' + os.path.join(allesfitter_path,'GUI.ipynb') + '"')

#::: version
__version__ = '0.3.0'