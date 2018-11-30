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
"""



import allesfitter



'''
uncomment whatever you want to run
'''


###############################################################################
#::: run the fit
###############################################################################
#allesfitter.ns_fit('allesfit_spots')
#allesfitter.ns_output('allesfit_spots')

allesfitter.ns_fit('allesfit_spots_fast')
allesfitter.ns_output('allesfit_spots_fast')