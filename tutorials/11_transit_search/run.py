#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:57:05 2018

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
from allesfitter import read_csv
from allesfitter.detection.transit_search import tls_search


'''
Values you will need:
-----------------------------------------------
time, flux, flux_err -- load from SPECULOOS pipeline lightcurves

Optional values that will help a better search:
-----------------------------------------------
R_star, R_star_lerr, R_star_uerr -- stellar radius and its uncertainty, load from TICv8 catalog 
M_star, M_star_lerr, M_star_uerr -- stellar mass and its uncertainty, load from TICv8 catalog 
u = [float, float] -- quadratic limb darkening, load from Claret tables or the like  
'''

time, flux, flux_err = read_csv('Leonardo.csv') #replace with your light curve
R_star, R_star_lerr, R_star_uerr = 1, 0.1, 0.1 #in Rsun, replace with your values
M_star, M_star_lerr, M_star_uerr = 1, 0.1, 0.1 #in Msun, replace with your values
u = [0.6, 0.4] #replace with your values

wotan_kwargs = {'slide_clip':{}, 'flatten':{}}
wotan_kwargs['slide_clip']['window_length'] = 1 #usually a good guess
wotan_kwargs['slide_clip']['low'] = 20
wotan_kwargs['slide_clip']['high'] = 3 #remove flares
wotan_kwargs['flatten']['method'] = 'biweight'  #or rspline
wotan_kwargs['flatten']['window_length'] = 1 #usually a good guess  

tls_kwargs = {}
tls_kwargs['R_star']=float(R_star)
tls_kwargs['R_star_min']=R_star-3*R_star_lerr
tls_kwargs['R_star_max']=R_star+3*R_star_uerr
tls_kwargs['M_star']=float(M_star)
tls_kwargs['M_star_min']=M_star-3*M_star_lerr
tls_kwargs['M_star_max']=M_star+3*M_star_uerr
tls_kwargs['u']=u   
tls_kwargs['SNR_threshold'] = 5.
tls_kwargs['SDE_threshold'] = 5.
tls_kwargs['FAP_threshold'] = 0.05

options = {}
options['show_plot'] = True #amend as you wish
options['save_plot'] = True #amend as you wish
options['outdir'] = 'lc_'+wotan_kwargs['flatten']['method']+'_window='+str(wotan_kwargs['flatten']['window_length']) #replace with your dream file name

tls_search(time, flux, flux_err,
            wotan_kwargs=wotan_kwargs,
            tls_kwargs=tls_kwargs,
            options=options)
    
    