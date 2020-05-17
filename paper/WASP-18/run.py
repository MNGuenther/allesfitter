#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:57:05 2018

@author:
Maximilian N. Günther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as c
import allesfitter
from allesfitter.v2.translator import translate




###############################################################################
#::: calculate prior on q (mass ratio) via Stassun+2017
###############################################################################
#Mp = np.random.normal(loc=(11.44*c.M_jup/c.M_sun).value, scale=(1.51*c.M_jup/c.M_sun).value, size=10000)
#Ms = np.random.normal(loc=1.46, scale=0.29, size=10000)
#q = Mp / Ms
#print( np.percentile(q, [16,50,84] ) )



###############################################################################
#::: translate S19 to allesfitter
###############################################################################
# translate({'a/R_host':3.562, 'R_companion/R_host':0.09716})



###############################################################################
#::: run the fit
###############################################################################
# allesfitter.show_initial_guess('allesfit_sine_physical')
# allesfitter.mcmc_fit('allesfit_sine_physical')
# allesfitter.mcmc_output('allesfit_sine_physical')

# allesfitter.show_initial_guess('allesfit_sine_series')
allesfitter.mcmc_fit('allesfit_sine_series')
allesfitter.mcmc_output('allesfit_sine_series')