#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 01:13:40 2018

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
 
import matplotlib.pyplot as plt
import allesfitter
from allesfitter.spots import plot_spots_from_posteriors, plot_publication_spots_from_posteriors

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




# allesfitter.show_initial_guess('allesfit_0')
# allesfitter.ns_fit('allesfit_0')
# allesfitter.ns_output('allesfit_0')

# allesfitter.show_initial_guess('allesfit_1')
# allesfitter.ns_fit('allesfit_1')
# allesfitter.ns_output('allesfit_1')

# allesfitter.show_initial_guess('allesfit_2')
# allesfitter.ns_fit('allesfit_2')
# allesfitter.ns_output('allesfit_2')