#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 17:15:01 2018

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

#::: modules
import numpy as np
from lichtkurven.index_transits import index_transits




###############################################################################
#::: cut away the out-of-transit regions to speed up the fit
###############################################################################   
def reduce_phot_data(time, flux, flux_err, params, settings):
    ind_in = []
          
    for planet in settings['planets_phot']:
        start = np.nanmin( time )
        first_epoch = 1.*params[planet+'_epoch']
        period      = 1.*params[planet+'_period']
        
        #::: place the first_epoch at the start of the data to avoid luser mistakes
        if start<=first_epoch:
            first_epoch -= int(np.round((first_epoch-start)/period)) * period
        else:
            first_epoch += int(np.round((start-first_epoch)/period)) * period
  
        dic = {'TIME':time, 'EPOCH':first_epoch, 'PERIOD':period, 'WIDTH':8./24.}
        ind_in += list(index_transits(dic)[0])
    time = time[ind_in]
    flux = flux[ind_in]
    flux_err = flux_err[ind_in]
    return time, flux, flux_err
