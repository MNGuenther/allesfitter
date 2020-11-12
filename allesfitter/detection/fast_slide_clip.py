#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:44:29 2020

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
import pandas as pd
from tqdm import tqdm
from glob import glob
from pprint import pprint
from wotan import flatten
from astropy.stats import sigma_clip

#::: my modules
import allesfitter

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




def fast_slide_clip(time, flux, window_length=1, low=5, high=5, return_mask=False):
    """
    A much faster alternative to Wotan's build-in slide clip

    Returns
    -------
    Clipped flux (outliers replaced with NaN)
    """
    flux2 = 1.*flux
    flux_flat = flatten(time, flux, method='biweight', window_length=window_length)
    flux_flat = sigma_clip(np.ma.masked_invalid(flux_flat), sigma_lower=low, sigma_upper=high) #astropy wants masked arrays
    flux2[flux_flat.mask] = np.nan #use NaN instead of masked arrays, because masked arrays drive me crazy
    
    if not return_mask:
        return flux2
    
    else: 
        with np.testing.suppress_warnings() as sup:
            sup.filter(UserWarning)
            mask_upper = (flux_flat > np.median(flux_flat)) * flux_flat.mask
            mask_lower = (flux_flat < np.median(flux_flat)) * flux_flat.mask
        return flux2, flux_flat.mask, mask_upper, mask_lower