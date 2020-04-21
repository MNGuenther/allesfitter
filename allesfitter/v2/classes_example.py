#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:49:37 2020

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
from tqdm import tqdm
from pprint import pprint

#::: my modules
from classes import allesclass2
from translator import translate

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})


#::: user settings
rr = 0.1
rsuma = 0.1
epoch = 0
period = 1
cosi = 0
r_host = 1
m_host = 1
time = np.linspace(-0.5,0.5,1001)

#::: allesclass2
alles = allesclass2()
alles.settings = {'companions_phot':['b'], 'inst_phot':['buffer']}
alles.params = {'b_rr':rr, 'b_rsuma':rsuma, 'b_epoch':epoch, 'b_period':period, 'b_cosi':cosi}
alles.params_host = {'R_host':r_host, 'M_host':m_host}
alles.fill()   

#::: model flux
model_flux = alles.generate_model('buffer', 'flux', time)
