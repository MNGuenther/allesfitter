#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:30:37 2020

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
import matplotlib.pyplot as plt    
from pprint import pprint
import allesfitter
import seaborn as sns



alles = allesfitter.allesclass('allesfit')
fig, axes = plt.subplots(2,1,figsize=(6,6),tight_layout=True)
colors = {'b': sns.color_palette()[0],
          'c': sns.color_palette()[1]}

def get_infos(companion):
    infos = {}
    
    infos['transit_midtime_linear'] = np.array( alles.data[companion+'_tmid_observed_transits'] ) #linear-ephemerides midtimes that fall into the observation windows
    infos['transit_number'] = np.array( [ int(np.round( ( t - alles.posterior_params_median[companion+'_epoch'] ) / alles.posterior_params_median[companion+'_period'] ) ) for t in infos['transit_midtime_linear']] )
    infos['N_transits'] = len(infos['transit_number']) #number of linear-ephemerides midtimes that fall into the observation windows
    
    infos['ttv_median'] = np.array( [alles.posterior_params_median[companion+'_ttv_transit_'+str(i+1)] for i in range(infos['N_transits'])] )
    infos['ttv_lerr'] = np.array( [alles.posterior_params_ll[companion+'_ttv_transit_'+str(i+1)] for i in range(infos['N_transits'])] )
    infos['ttv_uerr'] = np.array( [alles.posterior_params_ul[companion+'_ttv_transit_'+str(i+1)] for i in range(infos['N_transits'])] )

    infos['ttv_median'][infos['ttv_median']==0] = np.nan #remove the one we fixed
    infos['ttv_lerr'][infos['ttv_lerr']==0] = np.nan #remove the one we fixed
    infos['ttv_uerr'][infos['ttv_uerr']==0] = np.nan #remove the one we fixed
    
    infos['transit_midtime_median'] = infos['transit_midtime_linear'] + infos['ttv_median']
    infos['transit_midtime_lerr'] = infos['ttv_lerr']
    infos['transit_midtime_uerr'] = infos['ttv_uerr']
    
    k = ~np.isnan(infos['ttv_median'])
    z = np.polyfit(infos['transit_number'][k], infos['transit_midtime_median'][k], 1)
    p = np.poly1d(z)
    infos['period_linear'] = z[0]
    infos['epoch_linear'] = z[1]
    infos['o_minus_c_median'] = infos['transit_midtime_median'] - p(infos['transit_number'])
    infos['o_minus_c_lerr'] = infos['ttv_lerr']
    infos['o_minus_c_uerr'] = infos['ttv_uerr']
    
    return infos


def save_infos(companion, ax):
    infos = get_infos(companion)
    
    #::: save csv
    header = 'period_linear = '+str(infos['period_linear'])+'\n'+\
             'epoch_linear = '+str(infos['epoch_linear'])+'\n'+\
             'transit_number,transit_midtime_median,transit_midtime_lerr,transit_midtime_uerr,o_minus_c,o_minus_c_lerr,o_minus_c_uerr'
    X = np.column_stack((infos['transit_number'], infos['transit_midtime_median'], infos['transit_midtime_lerr'], infos['transit_midtime_uerr'], infos['o_minus_c_median'], infos['o_minus_c_lerr'], infos['o_minus_c_uerr']))
    np.savetxt('TOI-216_'+companion+'_summary.csv', X, delimiter=',', header=header)
    
    #::: save latex table
    with open('TOI-216_'+companion+'_latex_table.txt','w') as f:
        f.write('Transit mid-time ($\mathrm{BJD_\{TDB}$) & O-C (min.)\n')
        for i in range(infos['N_transits']):
            a = allesfitter.utils.latex_printer.round_tex(infos['transit_midtime_median'][i], infos['transit_midtime_lerr'][i], infos['transit_midtime_uerr'][i])
            b = allesfitter.utils.latex_printer.round_tex(infos['o_minus_c_median'][i], infos['o_minus_c_lerr'][i], infos['o_minus_c_uerr'][i])
            f.write('$'+a+'$' + ' & ' + '$'+b+'$' + '\\\\\n')
    
    #::: save plot
    # ax.errorbar(infos['transit_number'], infos['ttv_median']*24*60, yerr=[infos['ttv_lerr']*24*60,infos['ttv_uerr']*24*60], marker='o', ls='none', alpha=0.3, color=colors[companion]) #if you want to show the fitted TTVs (dependent on the initial guess epoch and period) instead of the O-C (which was linearly fit from the posteriors)
    ax.errorbar(infos['transit_number'], infos['o_minus_c_median']*24*60, yerr=[infos['o_minus_c_lerr']*24*60,infos['o_minus_c_uerr']*24*60], marker='o', ls='none', color=colors[companion])
    ax.axhline(0,c='grey',ls='--')
    ax.text(0.95,0.95,'TOI-216 '+companion,va='top',ha='right',transform=ax.transAxes)
    ax.set(ylabel='O-C (min)', xlabel='Transit Nr.')
    
    
save_infos('b',axes[0])
save_infos('c',axes[1])
fig.savefig('TOI-216_o_minus_c.pdf', bbox_innches='tight')
