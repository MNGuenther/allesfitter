#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:36:21 2019

@author:
Dr. Maximilian N. Günther
European Space Agency (ESA)
European Space Research and Technology Centre (ESTEC)
Keplerlaan 1, 2201 AZ Noordwijk, The Netherlands
Email: maximilian.guenther@esa.int
GitHub: mnguenther
Twitter: m_n_guenther
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
import os

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




def print_priors(datadir, typ='uniform', scaling=5):

    if os.path.exists( os.path.join(datadir,'results/ns_table.csv') ):
        f = os.path.join(datadir,'results/ns_table.csv')
    else:
        f = os.path.join(datadir,'results/mcmc_table.csv')
        
    results = np.genfromtxt( f, delimiter=',', dtype=('U100',float,float,float,'U100','U100'), encoding='utf-8', names=True )

    if typ=='uniform':
        for i in range(len(results['name'])):
            if np.isnan(results['lower_error'][i]):
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',0,,' + results['label'][i] + ',' + results['unit'][i])
            elif ('_period' in results['name'][i]) or ('_epoch' in results['name'][i]) or ('_slope' in results['name'][i]) or ('_offset' in results['name'][i]):
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',1,uniform ' + str(results['median'][i] - scaling*results['lower_error'][i]) + ' ' + str(results['median'][i] + scaling*results['upper_error'][i])  + ',' + results['label'][i] + ',' + results['unit'][i])
            elif ('_f_c' in results['name'][i]) or ('_f_s' in results['name'][i]):
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',1,uniform ' + str( np.max( [-1, results['median'][i] - scaling*results['lower_error'][i]] ) ) + ' ' + str( np.min( [1, results['median'][i] + scaling*results['upper_error'][i]] ) )  + ',' + results['label'][i] + ',' + results['unit'][i])
            elif ('baseline_gp' in results['name'][i]):
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',1,uniform ' + str( np.max( [-15, results['median'][i] - scaling*results['lower_error'][i]] ) ) + ' ' + str( np.min( [15, results['median'][i] + scaling*results['upper_error'][i]] ) )  + ',' + results['label'][i] + ',' + results['unit'][i])
            elif ('ln_err' in results['name'][i]) or ('ln_jitter' in results['name'][i]):
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',1,uniform ' + str( np.max( [-15, results['median'][i] - scaling*results['lower_error'][i]] ) ) + ' ' + str( np.min( [0, results['median'][i] + scaling*results['upper_error'][i]] ) )  + ',' + results['label'][i] + ',' + results['unit'][i])
            else:
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',1,uniform ' + str( np.max( [0, results['median'][i] - scaling*results['lower_error'][i]] ) ) + ' ' + str( np.min( [1, results['median'][i] + scaling*results['upper_error'][i]] ) )  + ',' + results['label'][i] + ',' + results['unit'][i])

    if typ=='trunc_normal':
        for i in range(len(results['name'])):
            if np.isnan(results['lower_error'][i]):
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',0,,' + results['label'][i] + ',' + results['unit'][i])
            elif ('_period' in results['name'][i]) or ('_epoch' in results['name'][i]) or ('_slope' in results['name'][i]) or ('_offset' in results['name'][i]):
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',1,trunc_normal 0 1e15 ' + str(results['median'][i]) + ' ' + scaling*str(np.max([results['lower_error'][i],results['upper_error'][i]])) + ',' + results['label'][i] + ',' + results['unit'][i])
            elif ('_f_c' in results['name'][i]) or ('_f_s' in results['name'][i]):
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',1,trunc_normal -1 1 ' + str(results['median'][i]) + ' ' + scaling*str(np.max([results['lower_error'][i],results['upper_error'][i]])) + ',' + results['label'][i] + ',' + results['unit'][i])
            elif ('baseline_gp' in results['name'][i]):
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',1,trunc_normal -15 15 ' + str(results['median'][i]) + ' ' + scaling*str(np.max([results['lower_error'][i],results['upper_error'][i]])) + ',' + results['label'][i] + ',' + results['unit'][i])
            elif ('ln_err' in results['name'][i]) or ('ln_jitter' in results['name'][i]):
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',1,trunc_normal -15 0 ' + str(results['median'][i]) + ' ' + scaling*str(np.max([results['lower_error'][i],results['upper_error'][i]])) + ',' + results['label'][i] + ',' + results['unit'][i])
            else:
                print(str(results['name'][i]) + ',' + str(results['median'][i]) + ',1,trunc_normal 0 1 ' + str(results['median'][i]) + ' ' + scaling*str(np.max([results['lower_error'][i],results['upper_error'][i]])) + ',' + results['label'][i] + ',' + results['unit'][i])

