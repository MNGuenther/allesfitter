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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches

#::: my modules
import allesfitter
from allesfitter.spots import plot_spots_from_posteriors, plot_spots_new

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




############################################################
#::: plot the fit
############################################################
# alles = allesfitter.allesclass('allesfit_1')
# fig, axes = plt.subplots(2, 1, figsize=(15,5), gridspec_kw={'height_ratios': [3,1]}, sharex='col')
# axes = np.atleast_2d(axes).T

# alles.plot('TESS','b','full',ax=axes[0,0], dt=0.5/24./60., kwargs_data={'rasterized':False}, kwargs_ax={'title':''})
# alles.plot('TESS','b','full_residuals',ax=axes[1,0], kwargs_data={'rasterized':False}, kwargs_ax={'title':''})
# axes[0,0].set(xlim=[alles.data['TESS']['time'][0],alles.data['TESS']['time'][-1]])

# rect = patches.Rectangle((2456519.02,0.989),0.03,0.105,linewidth=1,edgecolor='k',facecolor='none')
# axes[0,0].add_patch(rect)
# axins = inset_axes(axes[0,0], width=2.5, height=1.5, bbox_to_anchor=(.08, .5, .3, .5), bbox_transform=axes[0,0].transAxes)
# alles.plot('TESS','b','full',ax=axins, dt=0.5/24./60., kwargs_data={'rasterized':False}, kwargs_ax={'title':''})
# axins.set(xticks=[], yticks=[], xlabel='', ylabel='', xlim=[2456519.02,2456519.05] )

# rect = patches.Rectangle((2456519.805,0.989),0.03,0.04,linewidth=1,edgecolor='k',facecolor='none')
# axes[0,0].add_patch(rect)
# axins = inset_axes(axes[0,0], width=2.5, height=1.5, bbox_to_anchor=(.53, .5, .3, .5), bbox_transform=axes[0,0].transAxes)
# alles.plot('TESS','b','full',ax=axins, dt=0.5/24./60., kwargs_data={'rasterized':False}, kwargs_ax={'title':''})
# axins.set(xticks=[], yticks=[], xlabel='', ylabel='', xlim=[2456519.805,2456519.835], ylim=[0.989,0.989+0.04] )

# plt.tight_layout()
# fig.subplots_adjust(hspace=0)
# fig.savefig('GJ_1243_fit.pdf')




############################################################
#::: plot the spot maps
############################################################
# plot_spots_from_posteriors('allesfit_1', command='show', Nsamples=1)
plot_spots_new('allesfit_1')


############################################################
#::: plot the Bayes factors
############################################################
# sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=2.5, color_codes=True)
# fig, ax = plt.subplots(figsize=(20,6), tight_layout=True)
# allesfitter.ns_plot_bayes_factors(['allesfit_0','allesfit_1','allesfit_2','allesfit_0_free_cosi','allesfit_1_free_cosi','allesfit_2_free_cosi'], 
#                                   labels=['1 spot, 3 flares', '2 spots, 3 flares', '2 spots, 2 flares','1 spot, 3 flares, free cosi', '2 spots, 3 flares, free cosi', '2 spots, 2 flares, free cosi'], ax=ax)
# ax.set(yscale='log',ylim=[1,400])
# fig.savefig('GJ_1243_Bayes_factors.pdf', bbox_inchs='tight')


# sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=2.5, color_codes=True)
# fig, ax = plt.subplots(figsize=(10,6), tight_layout=True)
# allesfitter.ns_plot_bayes_factors(['allesfit_0','allesfit_1','allesfit_2'], 
#                                   labels=['1 spot, 3 flares', '2 spots, 3 flares', '2 spots, 2 flares'], ax=ax)
# ax.set(yscale='log',ylim=[1,ax.get_ylim()[1]])
# fig.savefig('GJ_1243_Bayes_factors.pdf', bbox_inchs='tight')