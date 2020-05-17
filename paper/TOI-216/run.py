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
Web: www.mnguenther.com
"""

import numpy as np
import allesfitter
from allesfitter.v2.translator import translate




###############################################################################
#::: translate D19 to allesfitter
###############################################################################
# translate({'a_over_R_host':29.1, 'rr':0.11, 'incl':88}) #planet b 
# translate({'a_over_R_host':53.8, 'rr':0.1236, 'incl':89.89}) #planet c



###############################################################################
#::: estimate mean epoch and period
###############################################################################
# def estimate_mean_epoch_and_period(period_approx, first_transit, last_transit):
#     N = int(np.round( (last_transit - first_transit) / period_approx ))
#     period_mean = 1.*(last_transit - first_transit)/N
#     epoch_mean = first_transit + period_mean * int(np.round(N/2.))
#     print(np.format_float_positional(period_mean,4), np.format_float_positional(epoch_mean,4))

# estimate_mean_epoch_and_period(17.1, 2458325.32, 2458666.85) #planet b
# estimate_mean_epoch_and_period(34.6, 2458331.28, 2458676.82) #planet c



###############################################################################
#::: TTV fit
###############################################################################
# allesfitter.prepare_ttv_fit('allesfit')
# allesfitter.estimate_noise_out_of_transit('allesfit')
# allesfitter.show_initial_guess('allesfit', kwargs_dict={'window':12./24.})
# allesfitter.ns_fit('allesfit')
# allesfitter.ns_output('allesfit')
