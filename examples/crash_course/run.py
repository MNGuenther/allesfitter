#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:57:05 2018

@author:
Maximilian N. Guenther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
"""


import allesfitter
reload(allesfitter)


'''
uncomment whichever model you want to run
'''

#allesfitter.ns_fit('allesfit_Leonardo_unif')
#allesfitter.ns_output('allesfit_Leonardo_unif')


#allesfitter.ns_fit('allesfit_Leonardo_rwalk')
#allesfitter.ns_output('allesfit_Leonardo_rwalk')


#allesfitter.ns_fit('allesfit_all_TMNT_unif')
#allesfitter.ns_output('allesfit_all_TMNT_unif')


#allesfitter.ns_fit('allesfit_all_TMNT_rwalk')
#allesfitter.ns_output('allesfit_all_TMNT_rwalk')


#allesfitter.ns_fit('allesfit_all_TMNT_rslice')
#allesfitter.ns_output('allesfit_all_TMNT_rslice')


allesfitter.ns_fit('allesfit_all_TMNT_hslice')
allesfitter.ns_output('allesfit_all_TMNT_hslice')