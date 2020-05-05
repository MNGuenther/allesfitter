#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:08:46 2020

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

#::: modules
import os, sys
import numpy as np
import matplotlib.pyplot as plt

#::: my modules
import allesfitter

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




###############################################################################
#::: plot the fit
###############################################################################
# alles = allesfitter.allesclass('allesfit_4')
# fig, axes = plt.subplots(2, 3, figsize=(15,5), gridspec_kw={'height_ratios': [3,1]}, sharex='col')

# alles.plot('TESS','c','phasezoom',ax=axes[0,0], kwargs_data={'rasterized':False}, kwargs_ax={'title':''})
# alles.plot('TESS','c','phasezoom_residuals',ax=axes[1,0], kwargs_data={'rasterized':False}, kwargs_ax={'title':''})
# axes[0,0].text(0.65,0.9,'Pi Mensae c', transform=axes[0,0].transAxes)
# axes[0,0].set(ylim=[0.9993,1.0005])
# axes[0,0].get_yaxis().get_major_formatter().set_useOffset(False)

# alles.plot('AAT','b','phase',ax=axes[0,1], kwargs_data={'color':'c', 'alpha':0.33, 'rasterized':False}, kwargs_ax={'title':''}, Nsamples=7)
# alles.plot('AAT','b','phase_residuals',ax=axes[1,1], kwargs_data={'color':'c', 'alpha':0.33, 'rasterized':False}, kwargs_ax={'title':''})
# alles.plot('HARPS1','b','phase',ax=axes[0,1], kwargs_data={'color':'b', 'marker':'.', 'rasterized':False}, kwargs_ax={'title':''}, Nsamples=7)
# alles.plot('HARPS1','b','phase_residuals',ax=axes[1,1], kwargs_data={'color':'b', 'marker':'.', 'rasterized':False}, kwargs_ax={'title':''})
# alles.plot('HARPS2','b','phase',ax=axes[0,1], kwargs_data={'color':'b', 'marker':'^', 'markersize':5, 'rasterized':False}, kwargs_ax={'title':''}, Nsamples=6)
# alles.plot('HARPS2','b','phase_residuals',ax=axes[1,1], kwargs_data={'color':'b', 'marker':'^', 'markersize':5, 'rasterized':False}, kwargs_ax={'title':''})
# axes[0,1].text(0.65,0.9,'Pi Mensae b', transform=axes[0,1].transAxes)

# alles.plot('AAT','c','phase',ax=axes[0,2], kwargs_data={'color':'c', 'alpha':0.33, 'rasterized':False}, kwargs_ax={'title':''}, Nsamples=7)
# alles.plot('AAT','c','phase_residuals',ax=axes[1,2], kwargs_data={'color':'c', 'alpha':0.33, 'rasterized':False}, kwargs_ax={'title':''})
# alles.plot('HARPS1','c','phase',ax=axes[0,2], kwargs_data={'color':'b', 'marker':'.', 'rasterized':False}, kwargs_ax={'title':''}, Nsamples=7)
# alles.plot('HARPS1','c','phase_residuals',ax=axes[1,2], kwargs_data={'color':'b', 'marker':'.', 'rasterized':False}, kwargs_ax={'title':''})
# alles.plot('HARPS2','c','phase',ax=axes[0,2], kwargs_data={'color':'b', 'marker':'^', 'markersize':5, 'rasterized':False}, kwargs_ax={'title':''}, Nsamples=6)
# alles.plot('HARPS2','c','phase_residuals',ax=axes[1,2], kwargs_data={'color':'b', 'marker':'^', 'markersize':5, 'rasterized':False}, kwargs_ax={'title':''})
# axes[0,2].text(0.65,0.9,'Pi Mensae c', transform=axes[0,2].transAxes)

# plt.tight_layout()
# fig.subplots_adjust(hspace=0)
# fig.savefig('Pi_Mensae_fit.pdf')



###############################################################################
#::: plot the Bayes factors
###############################################################################
# sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=2.5, color_codes=True)
# fig, ax = plt.subplots(figsize=(10,6), tight_layout=True)
# allesfitter.ns_plot_bayes_factors(['allesfit_1','allesfit_2','allesfit_3','allesfit_4'], 
#                                   labels=['standard','linear LD','eccentric','GP baseline'], ax=ax)
# ax.set(xticklabels=['linear LD\nvs.\nquadratic LD', 'eccentric\nvs.\ncircular', 'GP baseline\nvs.\nconstant baseline',])
# fig.savefig('Pi_Mensae_Bayes_factors.pdf', bbox_inchs='tight')



###############################################################################
#::: plot the posterior histograms between Sector 1 and Year 1
###############################################################################
allesfitter.plot_histograms(datadirs=['allesfit_4', 'allesfit_year_1_noshiftepoch'], 
                            titles=['Sector 1', 'Year 1'], 
                            keys=['c_rr', 'c_rsuma', 'c_cosi', 'c_epoch', 'c_period', 'host_ldc_q1_TESS', 'host_ldc_q2_TESS'], 
                            options={'layout':'2x4', 'simplify':True, 'colors':['lightgrey','b'], 'epoch_median':True, 'show_plot':True, 'save_plot':True})