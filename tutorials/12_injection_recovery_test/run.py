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
from allesfitter.transit_search.injection_recovery import inject_and_tls_search



time, flux, flux_err = np.genfromtxt('Leonardo.csv', delimiter=',', unpack=True)
periods = [1,2,3,4,5] #list of injection periods in days
rplanets = [1,2,3,4,5] #list of injection rplanets in Rearth
logfname = 'injection_recovery_test.csv'

inject_and_tls_search(time, flux, flux_err, 
                      periods, rplanets, logfname, 
                      SNR_threshold=5.)