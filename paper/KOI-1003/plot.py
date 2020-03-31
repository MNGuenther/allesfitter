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




############################################################
#::: plot the fit
############################################################
alles = allesfitter.allesclass('allesfit')

fig = plt.figure(figsize=(12,6), tight_layout=True)
gs = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[1,2])

alles.plot('Kepler','B','full',ax=ax1, kwargs_data={'rasterized':False}, kwargs_ax={'title':''})
alles.plot('Kepler','B','phase',ax=ax2, kwargs_data={'rasterized':False}, kwargs_ax={'title':''})
alles.plot('Kepler','B','phasezoom',ax=ax3, zoomwindow=16., kwargs_data={'rasterized':False}, kwargs_ax={'title':''})
alles.plot('Kepler','B','phasezoom_occ',ax=ax4, zoomwindow=16., kwargs_data={'rasterized':False}, kwargs_ax={'title':''})
ax3.set(ylabel='',yticks=[])
ax4.set(ylabel='',yticks=[])

fig.savefig('KOI-1003_fit.pdf')