#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:46:38 2020

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

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from . import defaults, translator, generator
from .. import computer




class allesclass2():
    
    def __init__(self, datadir=None):
        if datadir is None:
            self.settings = {}
            self.params = {}
            self.params_host = {}
            self.data = {}
        else:
            pass #TODO
        
        
    #==========================================================================
    #::: fill gaps with defaults
    #==========================================================================
    def fill_settings(self):
        self.settings = defaults.fill_settings(self.settings)
        
    def fill_params(self):
        self.params = defaults.fill_params(self.params, self.settings)
        
    def fill_params_host(self):
        self.params_host = defaults.fill_params_host(self.params_host)
        
    def fill(self):
        self.fill_settings()
        self.fill_params()
        self.fill_params_host()
        
        
    #==========================================================================
    #::: example initializations
    #==========================================================================
    def init_hot_jupiter(self):
        self.settings = defaults.get_hot_jupiter_settings()
        self.params = defaults.get_hot_jupiter_params()
        self.params_host = defaults.get_hot_jupiter_params_host()
        self.fill()
        
        
    #==========================================================================
    #::: add stuff
    #==========================================================================
    def add_companion_phot(self, name):
        if 'companions_phot' not in self.settings: self.settings['companions_phot'] = []
        self.settings['companions_phot'] += [name]
        self.fill()
        
    def add_companion_rv(self, name):
        if 'companions_rv' not in self.settings: self.settings['companions_rv'] = []
        self.settings['companions_rv'] += [name]
        self.fill()
        
    def add_inst_phot(self, name):
        if 'inst_phot' not in self.settings: self.settings['inst_phot'] = []
        self.settings['inst_phot'] += [name]
        self.fill()
        
    def add_inst_rv(self, name):
        if 'inst_rv' not in self.settings: self.settings['inst_rv'] = []
        self.settings['inst_rv'] += [name]
        self.fill()
        
    def add_flare(self, name):
        pass #TODO
    
    def add_observation(self, inst, key, time, y, yerr):
        self.data[inst] = {}
        self.data[inst]['time'] = time
        self.data[inst][key] = y
        self.data[inst][key+'_err'] = yerr
        
        
    #==========================================================================
    #::: do stuff
    #==========================================================================
    def generate_model(self, xx, inst='telescope', key='flux'):
        params_ellc = translator.translate_alles_to_ellc(self.params, self.settings)
        model = computer.calculate_model(params_ellc, inst, key, xx=xx, settings=self.settings)
        return model
        
    # def plot_model(self, inst, key, xx):
    #     model = self.generate_model(time, inst, key)
    #     pass
        
    # def inject_flux_model(self, inst, **kwargs):
    #     # generator.inject_transit_model(self.data[inst]['time'], self.data[inst]['flux'], self.data[inst]['flux_err'], params=self.params, settings=self.settings, **kwargs)
    #     pass
        
        
# class host():
    
#     def __init__(self, params=None, settings=None):
#         self.params = {'R_host': R_host,
#                        'M_host': M_host}

    

# class companion():
    
#     def __init__(self, name):
#         self.name = name
#         self.params = {'R_companion_earth': R_companion_earth,
#                        'M_companion_earth': M_companion_earth}