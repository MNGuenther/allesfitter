#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 19:51:37 2018

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




###############################################################################
#::: expand flags
###############################################################################
def expand_flags(flag, n=4):
    maskleft = np.where(flag>0)[0]
    flag_new = np.zeros(len(flag),dtype=bool)
    for i in range(-n,n+1):
        thismask = maskleft+i
        for t in thismask:
            if t<len(flag_new): flag_new[t] = 1
    return flag_new
