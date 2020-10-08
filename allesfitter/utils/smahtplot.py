#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:27:01 2020

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
from matplotlib.gridspec import GridSpec
import pandas as pd
from tqdm import tqdm
from glob import glob
from pprint import pprint
from brokenaxes import brokenaxes

#::: my modules
import allesfitter

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




def smahtplot(time, flux, subplot_spec=None):
    
    if subplot_spec is None: plt.figure()
    
    ind0 = [0] + list(np.where(np.diff(time)>10)[0]+1) #start indices of data chunks
    ind1 = list(np.where(np.diff(time)>10)[0]) + [len(time)-1] #end indices of data chunks
    xlims = [ (time[i],time[j]) for i,j in zip(ind0,ind1) ]
    
    bax = brokenaxes(xlims=xlims, subplot_spec=subplot_spec)
    bax.plot(time, flux, 'b.')
    
    return bax
