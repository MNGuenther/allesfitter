#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:47:38 2018

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

import numpy as np
from to_precision import std_notation




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
    return int(np.floor(np.log10(np.abs(x)))) - int(np.floor(np.log10(np.abs(y))))
    
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
