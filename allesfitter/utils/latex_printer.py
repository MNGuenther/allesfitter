#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:47:38 2018

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
from .to_precision import std_notation




def round_to_2(x):
    if x==0:
        return x
    else:
        return round(x, -int(np.floor(np.log10(np.abs(x))))+1)

def round_to_reference(x, y):
    return round(x, -int(np.floor(np.log10(np.abs(y))))+1)

def str_digits(y):
    if np.abs(y)<1: return -int(np.floor(np.log10(np.abs(y))))+1
    else: return int(np.floor(np.log10(np.abs(y))))+1
    
def extra_digits(x,y):
    try:
        return int(np.floor(np.log10(np.abs(x)))) - int(np.floor(np.log10(np.abs(y))))
    except:
        return 0
    
def round_tex(x, err_low, err_up, mode=None):
    if np.isnan(x):
        return 'NaN'
    y = np.min(( np.abs(err_low), np.abs(err_up) ))
    digs = extra_digits(x,y) + 2
    if np.abs(err_low-err_up)/np.mean([err_low,err_up]) > 0.05:
        txt = std_notation(x,digs) + '_{-' + std_notation(err_low,2) + '}^{+' + std_notation(err_up,2) + '}'
    else:
        txt = std_notation(x,digs) + '\\pm' + std_notation( np.max(( np.abs(err_low), np.abs(err_up) )) ,2)
    if mode is None:
        return txt
    else:
        return txt, std_notation(mode,digs)

def round_txt_separately(x, err_low, err_up):
    if np.isnan(x):
        return 'NaN'
    y = np.min(( np.abs(err_low), np.abs(err_up) ))
    digs = extra_digits(x,y) + 2
    txt1 = std_notation(x,digs) 
    txt2 = std_notation(err_low,2) 
    txt3 = std_notation(err_up,2)
    return txt1, txt2, txt3