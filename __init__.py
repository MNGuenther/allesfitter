#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:18:20 2018

@author:
Dr. Maximilian N. Guenther
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

from .mcmc import mcmc_fit
from .nested_sampling import ns_fit
from .general_output import get_labels, get_initial_guess_gp
from .nested_sampling_output import get_ns_samples, ns_output
from .mcmc_output import get_mcmc_samples, mcmc_output

__version__ = '0.2.0'