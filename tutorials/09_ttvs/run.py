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



import allesfitter



#::: lienar ephemerides fit
# allesfitter.show_initial_guess('allesfit_linear_ephemerides')
# allesfitter.ns_fit('allesfit_linear_ephemerides')
# allesfitter.ns_output('allesfit_linear_ephemerides')



#::: TTV fit
# allesfitter.prepare_ttv_fit('allesfit_with_ttvs')
# allesfitter.show_initial_guess('allesfit_with_ttvs')
# allesfitter.ns_fit('allesfit_with_ttvs')
allesfitter.ns_output('allesfit_with_ttvs')
