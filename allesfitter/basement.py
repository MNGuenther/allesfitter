#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:17:06 2018

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
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import collections
from datetime import datetime
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 
from scipy.special import ndtri
from scipy.stats import truncnorm

#::: allesfitter modules
from .exoworlds_rdx.lightcurves.index_transits import index_transits, index_eclipses, get_first_epoch, get_tmid_observed_transits
from .priors.simulate_PDF import simulate_PDF

                     
    
    
###############################################################################
#::: 'Basement' class, which contains all the data, settings, etc.
###############################################################################
class Basement():
    '''
    The 'Basement' class contains all the data, settings, etc.
    '''
    
    ###############################################################################
    #::: init
    ###############################################################################
    def __init__(self, datadir):
        '''
        Inputs:
        -------
        datadir : str
            the working directory for allesfitter
            must contain all the data files
            output directories and files will also be created inside datadir
        fast_fit : bool (optional; default is False)
            if False: 
                use all photometric data for the plot
            if True: 
                only use photometric data in an 8h window around the transit 
                requires a good initial guess of the epoch and period
                
        Returns:
        --------
        All the variables needed for allesfitter
        '''
        
        self.now = datetime.now().isoformat()
        
        self.datadir = datadir
        
        self.load_settings()
        self.load_params()
        self.load_data()
        
        if self.settings['shift_epoch']:
            self.change_epoch()
        
        if self.settings['fit_ttvs']:  
            self.prepare_ttv_fit()
        
        #::: external priors (e.g. stellar density)
        self.external_priors = {}
        self.load_stellar_priors()
        
        #::: if baseline model == sample_GP, set up a GP object for photometric data
#        self.setup_GPs()
        
        #::: translate limb darkening codes from params.csv (int) into str for ellc
        self.ldcode_to_ldstr = ["none",#   :  0,
                                "lin",#    :  1,
                                "quad",#   :  2,
                                "sing",#   :  3,
                                "claret",# :  4,
                                "log",#  :  5,
                                "sqrt",#  :  6,
                                "exp",#    :  7,
                                "power-2",#:  8,
                                "mugrid"]# : -1
        
        
        #::: set up the outdir
        self.outdir = os.path.join(datadir,'results') 
        if not os.path.exists( self.outdir ): os.makedirs( self.outdir )

        #::: check if the input is consistent
        for inst in self.settings['inst_phot']:
            key='flux'
            if (self.settings['baseline_'+key+'_'+inst] in ['sample_GP_Matern32', 'sample_GP_SHO']) &\
               (self.settings['error_'+key+'_'+inst] != 'sample'):
                   raise ValueError('If you want to use '+self.settings['baseline_'+key+'_'+inst]+', you will want to sample the jitters, too!')
            
                 
                    
    ###############################################################################
    #::: print function that prints into console and logfile at the same time
    ############################################################################### 
    def logprint(self, *text):
        print(*text)
        original = sys.stdout
        with open( os.path.join(self.outdir,'logfile_'+self.now+'.log'), 'a' ) as f:
            sys.stdout = f
            print(*text)
        sys.stdout = original
        
        

    ###############################################################################
    #::: load settings
    ###############################################################################
    def load_settings(self):
        '''
        Below is a copy of a "complete" settings.csv file:
            
        ###############################################################################
        # General settings
        ###############################################################################
        companions_phot 
        companions_rv
        inst_phot
        inst_rv
        ###############################################################################
        # Fit performance settings
        ###############################################################################
        fast_fit                              : optional. Default is False.
        fast_fit_width                        : optional. Default is 8./24.
        secondary_eclipse                     : optional. Default is False.
        multiprocess                          : optional. Default is False.
        multiprocess_cores                    : optional. Default is cpu_count()-1.
        ###############################################################################
        # MCMC settings
        ###############################################################################      
        mcmc_pre_run_loops                    : optional. Default is 0.         
        mcmc_pre_run_steps                    : optional. Default is 0. 
        mcmc_nwalkers                         : optional. Default is 100.
        mcmc_total_steps                      : optional. Default is 2000.
        mcmc_burn_steps                       : optional. Default is 1000.
        mcmc_thin_by                          : optional. Default is 1.
        ###############################################################################
        # Nested Sampling settings
        ###############################################################################
        ns_modus                              : optional. Default is static.
        ns_nlive                              : optional. Default is 500.
        ns_bound                              : optional. Default is single.
        ns_sample                             : optional. Default is rwalk.
        ns_tol                                : optional. Default is 0.01.
        ###############################################################################
        # Exposure settings for interpolation
        # needs to be in the same units as the time series
        # if not given the observing times will not be interpolated leading to biased results
        ###############################################################################
        t_exp_Leonardo                        : optional. Default is None.
        t_exp_Michelangelo                    : optional. Default is None.
        t_exp_Donatello                       : optional. Default is None.
        t_exp_Raphael                         : optional. Default is None.
        ###############################################################################
        # Number of points for exposure interpolation
        # Sample as fine as possible; generally at least with a 2 min sampling for photometry
        # n_int=5 was found to be a good number of interpolation points for any short phot. cadence t_exp; 
        # increase to n_int=10 for 30 min phot. cadence
        # the impact on RV is not as drasctic and generally n_int=5 is fine enough
        ###############################################################################
        t_exp_n_int_Leonardo                  : optional. Default is None.
        t_exp_n_int_Michelangelo              : optional. Default is None.
        t_exp_n_int_Donatello                 : optional. Default is None.
        t_exp_n_int_Raphael                   : optional. Default is None.
        ###############################################################################
        # Limb darkening law per instrument: lin / quad / sing
        #if 'lin' one corresponding parameter called 'ldc_q1_inst' has to be given in params.csv
        #if 'quad' two corresponding parameter called 'ldc_q1_inst' and 'ldc_q2_inst' have to be given in params.csv
        #if 'sing' three corresponding parameter called 'ldc_q1_inst'; 'ldc_q2_inst' and 'ldc_q3_inst' have to be given in params.csv
        ###############################################################################
        ld_law_Leonardo                       : optional. Default is quad.
        ld_law_Michelangelo                   : optional. Default is quad.
        ###############################################################################
        # Baseline settings
        # baseline params per instrument: sample_offset / sample_linear / sample_GP / hybrid_offset / hybrid_poly_1 / hybrid_poly_2 / hybrid_poly_3 / hybrid_pol_4 / hybrid_pol_5 / hybrid_pol_6 / hybrid_spline / hybrid_GP
        # if 'sample_offset' one corresponding parameter called 'baseline_offset_key_inst' has to be given in params.csv
        # if 'sample_linear' two corresponding parameters called 'baseline_a_key_inst' and 'baseline_b_key_inst' have to be given in params.csv
        ###############################################################################
        baseline_flux_Leonardo                : optional. Default is 'hybrid_spline'.
        baseline_flux_Michelangelo            : optional. Default is 'hybrid_spline'.
        baseline_rv_Donatello                 : optional. Default is 'hybrid_offset'.
        baseline_rv_Raphael                   : optional. Default is 'hybrid_offset'.
        ###############################################################################
        # Error settings
        # errors (overall scaling) per instrument: sample / hybrid
        # if 'sample' one corresponding parameter called 'inv_sigma2_key_inst' has to be given in params.csv (note this must be 1/sigma^2; not sigma)
        ###############################################################################
        error_flux_TESS                       : optional. Default is 'sample'.
        error_rv_AAT                          : optional. Default is 'sample'.
        error_rv_Coralie                      : optional. Default is 'sample'.
        error_rv_FEROS                        : optional. Default is 'sample'.

        '''
        
        
        def set_bool(text):
            if text.lower() in ['true', '1']:
                return True
            else:
                return False
            
        def unique(array):
            uniq, index = np.unique(array, return_index=True)
            return uniq[index.argsort()]
            
        rows = np.genfromtxt( os.path.join(self.datadir,'settings.csv'),dtype=None,encoding='utf-8',delimiter=',' )

        #::: make backwards compatible
        for i, row in enumerate(rows):
#            print(row)
            name = row[0]
            if name[:7]=='planets':
                rows[i][0] = 'companions'+name[7:]
                warnings.warn('Deprecation warning. You are using outdated keywords. Automatically renaming '+name+' ---> '+rows[i][0])
            if name[:6]=='ld_law':
                rows[i][0] = 'host_ld_law'+name[6:]
                warnings.warn('Deprecation warning. You are using outdated keywords. Automatically renaming '+name+' ---> '+rows[i][0])
                
#        self.settings = {r[0]:r[1] for r in rows}
        self.settings = collections.OrderedDict( [('user-given:','')]+[ (r[0],r[1] ) for r in rows ]+[('automatically set:','')] )

        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Main settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for key in ['companions_phot', 'companions_rv', 'inst_phot', 'inst_rv']:
            if key not in self.settings:
                self.settings[key] = []
            elif len(self.settings[key]): 
                self.settings[key] = str(self.settings[key]).split(' ')
            else:                       
                self.settings[key] = []
        
        self.settings['companions_all']  = list(np.unique(self.settings['companions_phot']+self.settings['companions_rv'])) #sorted by b, c, d...
        self.settings['inst_all'] = list(unique( self.settings['inst_phot']+self.settings['inst_rv'] )) #sorted like user input
    
    
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: General settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'print_progress' in self.settings:
            self.settings['print_progress'] = set_bool(self.settings['print_progress'] )
        else:
            self.settings['print_progress'] = True
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Epoch settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'shift_epoch' in self.settings:
            self.settings['shift_epoch'] = set_bool(self.settings['shift_epoch'] )
        else:
            self.settings['shift_epoch'] = False
            
            
        for companion in self.settings['companions_all']:
            if 'inst_for_'+companion+'_epoch' not in self.settings:
                self.settings['inst_for_'+companion+'_epoch'] = 'all'
        
            if self.settings['inst_for_'+companion+'_epoch'] in ['all','none']:
                self.settings['inst_for_'+companion+'_epoch'] = self.settings['inst_all']
            else:
                if len(self.settings['inst_for_'+companion+'_epoch']): 
                    self.settings['inst_for_'+companion+'_epoch'] = str(self.settings['inst_for_'+companion+'_epoch']).split(' ')
                else:                       
                    self.settings['inst_for_'+companion+'_epoch'] = []
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Multiprocess settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        self.settings['multiprocess'] = set_bool(self.settings['multiprocess'])
        
        if 'multiprocess_cores' not in self.settings.keys():
            self.settings['multiprocess_cores'] = cpu_count()-1
        elif self.settings['multiprocess_cores'] == 'all':
            self.settings['multiprocess_cores'] = cpu_count()-1
        else:
            self.settings['multiprocess_cores'] = int(self.settings['multiprocess_cores'])
            if self.settings['multiprocess_cores'] == cpu_count():
                string = 'You are pushing your luck: you want to run on '+str(self.settings['multiprocess_cores'])+' cores, but your computer has only '+str(cpu_count())+'. I will let you go through with it this time...'
                warnings.warn(string)
            if self.settings['multiprocess_cores'] > cpu_count():
                string = 'Oops, you want to run on '+str(self.settings['multiprocess_cores'])+' cores, but your computer has only '+str(cpu_count())+'. Maybe try running on '+str(cpu_count()-1)+'?'
                raise ValueError(string)


        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Phase variations
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('phase_variations' in self.settings.keys()) and len(self.settings['phase_variations']):
            self.settings['phase_variations'] = set_bool(self.settings['phase_variations'])
            if self.settings['phase_variations']==True:                
                self.logprint('The user set phase_variations==True. Automatically set fast_fit=False and secondary_eclispe=True, and overwrite other settings.')
                self.settings['fast_fit'] = 'False'
                self.settings['secondary_eclipse'] = 'True'
        else:
            self.settings['phase_variations'] = False
            
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Fast fit
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        self.settings['fast_fit'] = set_bool(self.settings['fast_fit'])
        
        if ('fast_fit_width' in self.settings.keys()) and len(self.settings['fast_fit_width']):
            self.settings['fast_fit_width'] = np.float(self.settings['fast_fit_width'])
        else:
            self.settings['fast_fit_width'] = 8./24.
                
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Host stellar density prior
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'use_host_density_prior' in self.settings:
            self.settings['use_host_density_prior'] = set_bool(self.settings['use_host_density_prior'] )
        else:
            self.settings['use_host_density_prior'] = True
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: TTVs
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('fit_ttvs' in self.settings.keys()) and len(self.settings['fit_ttvs']):
            self.settings['fit_ttvs'] = set_bool(self.settings['fit_ttvs'])
            if (self.settings['fit_ttvs']==True) and (self.settings['fast_fit']==False):
                raise ValueError("fit_ttvs==True, but fast_fit==False. Currently, you can only fit for TTVs if fast_fit==True. Please choose different settings.")
        else:
            self.settings['fit_ttvs'] = False
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Secondary eclipse
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('secondary_eclipse' in self.settings.keys()) and len(self.settings['secondary_eclipse']):
            self.settings['secondary_eclipse'] = set_bool(self.settings['secondary_eclipse'])
        else:
            self.settings['secondary_eclipse'] = False
                        
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: MCMC settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'mcmc_pre_run_loops' not in self.settings: 
            self.settings['mcmc_pre_run_loops'] = 0
        if 'mcmc_pre_run_steps' not in self.settings: 
            self.settings['mcmc_pre_run_steps'] = 0
        if 'mcmc_nwalkers' not in self.settings: 
            self.settings['mcmc_nwalkers'] = 100
        if 'mcmc_total_steps' not in self.settings: 
            self.settings['mcmc_total_steps'] = 2000
        if 'mcmc_burn_steps' not in self.settings: 
            self.settings['mcmc_burn_steps'] = 1000
        if 'mcmc_thin_by' not in self.settings: 
            self.settings['mcmc_thin_by'] = 1
                
        for key in ['mcmc_nwalkers','mcmc_pre_run_loops','mcmc_pre_run_steps','mcmc_total_steps','mcmc_burn_steps','mcmc_thin_by']:
            self.settings[key] = int(self.settings[key])
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Nested Sampling settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'ns_modus' not in self.settings: 
            self.settings['ns_modus'] = 'static'
        if 'ns_nlive' not in self.settings: 
            self.settings['ns_nlive'] = 500
        if 'ns_bound' not in self.settings: 
            self.settings['ns_bound'] = 'single'
        if 'ns_sample' not in self.settings: 
            self.settings['ns_sample'] = 'rwalk'
        if 'ns_tol' not in self.settings: 
            self.settings['ns_tol'] = 0.01
                
        self.settings['ns_nlive'] = int(self.settings['ns_nlive'])
        self.settings['ns_tol'] = float(self.settings['ns_tol'])
        
#        if self.settings['ns_sample'] == 'auto':
#            if self.ndim < 10:
#                self.settings['ns_sample'] = 'unif'
#                print('Using ns_sample=="unif".')
#            elif 10 <= self.ndim <= 20:
#                self.settings['ns_sample'] = 'rwalk'
#                print('Using ns_sample=="rwalk".')
#            else:
#                self.settings['ns_sample'] = 'slice'
#                print('Using ns_sample=="slice".')
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: host & companion grids, limb darkening laws, shapes
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for companion in self.settings['companions_all']:
            for inst in self.settings['inst_all']:
                
                if 'host_grid_'+inst not in self.settings: 
                    self.settings['host_grid_'+inst] = 'default'
                    
                if companion+'_grid_'+inst not in self.settings: 
                    self.settings[companion+'_grid_'+inst] = 'default'
                    
                if 'host_ld_law_'+inst not in self.settings or self.settings['host_ld_law_'+inst] is None or len(self.settings['host_ld_law_'+inst])==0 or self.settings['host_ld_law_'+inst]=='None': 
                    self.settings['host_ld_law_'+inst] = None
                    
                if companion+'_ld_law_'+inst not in self.settings or self.settings[companion+'_ld_law_'+inst] is None or len(self.settings[companion+'_ld_law_'+inst])==0 or self.settings[companion+'_ld_law_'+inst]=='None': 
                    self.settings[companion+'_ld_law_'+inst] = None
                
                if 'host_shape_'+inst not in self.settings: 
                    self.settings['host_shape_'+inst] = 'sphere'
                    
                if companion+'_shape_'+inst not in self.settings: 
                    self.settings[companion+'_shape_'+inst] = 'sphere'
                    
                    
        for companion in self.settings['companions_rv']:
            for inst in self.settings['inst_rv']:
                if companion+'_flux_weighted_'+inst in self.settings: 
                    self.settings[companion+'_flux_weighted_'+inst] = set_bool(self.settings[companion+'_flux_weighted_'+inst])
                else:
                    self.settings[companion+'_flux_weighted_'+inst] = False
        
                
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Stellar variability
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for key in ['flux', 'rv']:
            if ('stellar_var_'+key not in self.settings) or (self.settings['stellar_var_'+key] is None) or (self.settings['stellar_var_'+key].lower()=='none'): 
                self.settings['stellar_var_'+key] = 'none'
                     
                     
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Baselines
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_phot']:
            for key in ['flux']:
                if 'baseline_'+key+'_'+inst not in self.settings: 
                    self.settings['baseline_'+key+'_'+inst] = 'none'

                elif self.settings['baseline_'+key+'_'+inst] == 'sample_GP': 
                     warnings.warn('Deprecation warning. You are using outdated keywords. Automatically renaming sample_GP ---> sample_GP_Matern32.')
                     self.settings['baseline_'+key+'_'+inst] = 'sample_GP_Matern32'
                     
        for inst in self.settings['inst_rv']:
            for key in ['rv']:
                if 'baseline_'+key+'_'+inst not in self.settings: 
                    self.settings['baseline_'+key+'_'+inst] = 'none'
                    
                elif self.settings['baseline_'+key+'_'+inst] == 'sample_GP': 
                     warnings.warn('Deprecation warning. You are using outdated keywords. Automatically renaming sample_GP ---> sample_GP_Matern32.')
                     self.settings['baseline_'+key+'_'+inst] = 'sample_GP_Matern32'
                     
                
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Errors
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_phot']:
            for key in ['flux']:
                if 'error_'+key+'_'+inst not in self.settings: 
                    self.settings['error_'+key+'_'+inst] = 'sample'

        for inst in self.settings['inst_rv']:
            for key in ['rv']:
                if 'error_'+key+'_'+inst not in self.settings: 
                    self.settings['error_'+key+'_'+inst] = 'sample'
                    
                
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Color plot
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'color_plot' not in self.settings.keys():
            self.settings['color_plot'] = False
            
            
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Companion colors
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for i, companion in enumerate( self.settings['companions_all'] ):
            self.settings[companion+'_color'] = sns.color_palette()[i]
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Exposure time interpolation
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_all']:
            #::: if t_exp is given
            if 't_exp_'+inst in self.settings.keys() and len(self.settings['t_exp_'+inst]):
                t_exp = self.settings['t_exp_'+inst].split(' ')
                #if float
                if len(t_exp)==1:
                    self.settings['t_exp_'+inst] = np.float(t_exp[0])
                #if array
                else:
                    self.settings['t_exp_'+inst] = np.array([ np.float(t) for t in t_exp ])
            #::: if not given / given as an empty field
            else:
                self.settings['t_exp_'+inst] = None
                
            #::: if t_exp_n_int is given
            if 't_exp_'+inst in self.settings \
                and 't_exp_n_int_'+inst in self.settings \
                and len(self.settings['t_exp_n_int_'+inst]):
                    
                self.settings['t_exp_n_int_'+inst] = int(self.settings['t_exp_n_int_'+inst])
                if self.settings['t_exp_n_int_'+inst] < 1:
                    raise ValueError('"t_exp_n_int_'+inst+'" must be >= 1, but is given as '+str(self.settings['t_exp_n_int_'+inst])+' in params.csv')
            else:
                self.settings['t_exp_n_int_'+inst] = None
  
    
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Number of spots
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_all']:
            if 'host_N_spots_'+inst in self.settings and len(self.settings['host_N_spots_'+inst]):
                self.settings['host_N_spots_'+inst] = int(self.settings['host_N_spots_'+inst])
            else:
                self.settings['host_N_spots_'+inst] = 0
        
            for companion in self.settings['companions_all']:
                if companion+'_N_spots'+inst in self.settings:
                    self.settings[companion+'_N_spots_'+inst] = int(self.settings[companion+'_N_spots_'+inst])
                else:
                    self.settings[companion+'_N_spots_'+inst] = 0
                    
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Number of flares
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'N_flares' in self.settings and len(self.settings['N_flares'])>0:
            self.settings['N_flares'] = int(self.settings['N_flares'])
        else:
            self.settings['N_flares'] = 0
        
        
        
        
                
    ###############################################################################
    #::: load params
    ###############################################################################
    def load_params(self):
        '''
        #name	value	fit	bounds	label	unit
        #b_: companion name; _key : flux/rv/centd; _inst : instrument name					
        #dilution per instrument					
        dil_TESS	0	0	none	$D_\mathrm{TESS}$	
        dil_HATS	0.14	1	trunc_normal 0 1 0.14 0.1	$D_\mathrm{HATS}$	
        dil_FTS_i	0	0	none	$D_\mathrm{FTS_i}$	
        dil_GROND_g	0	0	none	$D_\mathrm{GROND_g}$	
        dil_GROND_r	0	0	none	$D_\mathrm{GROND_r}$	
        dil_GROND_i	0	0	none	$D_\mathrm{GROND_i}$	
        dil_GROND_z	0	0	none	$D_\mathrm{GROND_i}$	
        #limb darkening coefficients per instrument					
        ldc_q1_TESS	0.5	1	uniform 0 1	$q_{1;\mathrm{TESS}}$	
        ldc_q2_TESS	0.5	1	uniform 0 1	$q_{1;\mathrm{TESS}}$	
        ldc_q1_HATS	0.5	1	uniform 0 1	$q_{1;\mathrm{HATS}}$	
        ldc_q2_HATS	0.5	1	uniform 0 1	$q_{2;\mathrm{HATS}}$	
        ldc_q1_FTS_i	0.5	1	uniform 0 1	$q_{1;\mathrm{FTS_i}}$	
        ldc_q2_FTS_i	0.5	1	uniform 0 1	$q_{2;\mathrm{FTS_i}}$	
        ldc_q1_GROND_g	0.5	1	uniform 0 1	$q_{1;\mathrm{GROND_g}}$	
        ldc_q2_GROND_g	0.5	1	uniform 0 1	$q_{2;\mathrm{GROND_g}}$	
        ldc_q1_GROND_r	0.5	1	uniform 0 1	$q_{1;\mathrm{GROND_r}}$	
        ldc_q2_GROND_r	0.5	1	uniform 0 1	$q_{2;\mathrm{GROND_r}}$	
        ldc_q1_GROND_i	0.5	1	uniform 0 1	$q_{1;\mathrm{GROND_i}}$	
        ldc_q2_GROND_i	0.5	1	uniform 0 1	$q_{2;\mathrm{GROND_i}}$	
        ldc_q1_GROND_z	0.5	1	uniform 0 1	$q_{1;\mathrm{GROND_z}}$	
        ldc_q2_GROND_z	0.5	1	uniform 0 1	$q_{2;\mathrm{GROND_z}}$	
        #brightness per instrument per companion					
        b_sbratio_TESS	0	0	none	$J_{b;\mathrm{TESS}}$	
        b_sbratio_HATS	0	0	none	$J_{b;\mathrm{HATS}}$	
        b_sbratio_FTS_i	0	0	none	$J_{b;\mathrm{FTS_i}}$	
        b_sbratio_GROND_g	0	0	none	$J_{b;\mathrm{GROND_g}}$	
        b_sbratio_GROND_r	0	0	none	$J_{b;\mathrm{GROND_r}}$	
        b_sbratio_GROND_i	0	0	none	$J_{b;\mathrm{GROND_i}}$	
        b_sbratio_GROND_z	0	0	none	$J_{b;\mathrm{GROND_z}}$	
        #companion b astrophysical params					
        b_rsuma	0.178	1	trunc_normal 0 1 0.178 0.066	$(R_\star + R_b) / a_b$	
        b_rr	0.1011	1	trunc_normal 0 1 0.1011 0.0018	$R_b / R_\star$	
        b_cosi	0.099	1	trunc_normal 0 1 0.099 0.105	$\cos{i_b}$	
        b_epoch	2456155.967	1	trunc_normal 0 1e12 2456155.96734 0.00042 	$T_{0;b}$	$\mathrm{BJD}$
        b_period	3.547851	1	trunc_normal 0 1e12 3.547851 1.5e-5	$P_b$	$\mathrm{d}$
        b_K	0.1257	1	trunc_normal 0 1 0.1257 0.0471	$K_b$	$\mathrm{km/s}$
        b_q	1	0	none	$M_b / M_\star$	
        b_f_c	0	0	none	$\sqrt{e_b} \cos{\omega_b}$	
        b_f_s	0	0	none	$\sqrt{e_b} \sin{\omega_b}$	
        #TTVs					
        ...
        #Period changes					
        b_pv_TESS	0	0	trunc_normal -0.04 0.04 0 0.0007	$PV_\mathrm{TESS}$	$\mathrm{d}$
        b_pv_HATS	0	0	trunc_normal -0.04 0.04 0 0.0007	$PV_\mathrm{HATS}$	$\mathrm{d}$
        b_pv_FTS_i	0	0	trunc_normal -0.04 0.04 0 0.0007	$PV_\mathrm{FTS_i}$	$\mathrm{d}$
        b_pv_GROND_g	0	0	trunc_normal -0.04 0.04 0 0.0007	$PV_\mathrm{GROND_g}$	$\mathrm{d}$
        b_pv_GROND_r	0	0	trunc_normal -0.04 0.04 0 0.0007	$PV_\mathrm{GROND_r}$	$\mathrm{d}$
        b_pv_GROND_i	0	0	trunc_normal -0.04 0.04 0 0.0007	$PV_\mathrm{GROND_i}$	$\mathrm{d}$
        b_pv_GROND_z	0	0	trunc_normal -0.04 0.04 0 0.0007	$PV_\mathrm{GROND_i}$	$\mathrm{d}$
        #errors (overall scaling) per instrument					
        log_err_flux_TESS	-5.993	1	trunc_normal -23 0 -5.993 0.086	$\log{\sigma} (F_\mathrm{TESS})$	$\log{\mathrm{(rel. flux)}}$
        log_err_flux_HATS	-4.972	1	trunc_normal -23 0 -4.972 0.099	$\log{\sigma} (F_\mathrm{HATS})$	$\log{\mathrm{(rel. flux)}}$
        log_err_flux_FTS_i	-6	1	trunc_normal -23 0 -6.0 0.19	$\log{\sigma} (F_\mathrm{FTS_i})$	$\log{\mathrm{(rel. flux)}}$
        log_err_flux_GROND_g	-7.2	1	trunc_normal -23 0 -7.20 0.26	$\log{\sigma} (F_\mathrm{GROND_g})$	$\log{\mathrm{(rel. flux)}}$
        log_err_flux_GROND_r	-7.49	1	trunc_normal -23 0 -7.49 0.26	$\log{\sigma} (F_\mathrm{GROND_r})$	$\log{\mathrm{(rel. flux)}}$
        log_err_flux_GROND_i	-7.47	1	trunc_normal -23 0 -7.47 0.28	$\log{\sigma} (F_\mathrm{GROND_i})$	$\log{\mathrm{(rel. flux)}}$
        log_err_flux_GROND_z	-7.09	1	trunc_normal -23 0 -7.09 0.27	$\log{\sigma} (F_\mathrm{GROND_z})$	$\log{\mathrm{(rel. flux)}}$
        log_jitter_rv_AAT	-2.7	1	trunc_normal -23 0 -2.7 1.8	$\log{\sigma_\mathrm{jitter}} (RV_\mathrm{AAT})$	$\log{\mathrm{km/s}}$
        log_jitter_rv_Coralie	-2.7	1	trunc_normal -23 0 -2.7 1.5	$\log{\sigma_\mathrm{jitter}} (RV_\mathrm{Coralie})$	$\log{\mathrm{km/s}}$
        log_jitter_rv_FEROS	-5	1	trunc_normal -23 0 -5 15	$\log{\sigma_\mathrm{jitter}} (RV_\mathrm{FEROS})$	$\log{\mathrm{km/s}}$
        '''
        
        
    
        buf = np.genfromtxt(os.path.join(self.datadir,'params.csv'), delimiter=',',comments='#',dtype=None,encoding='utf-8',names=True)
        
        #::: make backwards compatible
        for i, name in enumerate(np.atleast_1d(buf['name'])):
            if name[:7]=='light_3':
                buf['name'][i] = 'dil_'+name[8:]
        
        for i, name in enumerate(np.atleast_1d(buf['name'])):
            if name[:3]=='ldc':
                buf['name'][i] = 'host_'+name
                
        #::: proceed...                    
        self.allkeys = np.atleast_1d(buf['name']) #len(all rows in params.csv)
        self.labels = np.atleast_1d(buf['label']) #len(all rows in params.csv)
        self.units = np.atleast_1d(buf['unit'])   #len(all rows in params.csv)
        if 'truth' in buf.dtype.names:
            self.truths = np.atleast_1d(buf['truth']) #len(all rows in params.csv)
        else:
            self.truths = np.nan * np.ones(len(self.allkeys))
            
            
        self.params = collections.OrderedDict()                                #len(all rows in params.csv)
        self.params['user-given:'] = ''                                        #just for printing
        for i,key in enumerate(self.allkeys):
            #::: if it's not a "coupled parameter", then use the given value
            if np.atleast_1d(buf['value'])[i] not in self.allkeys:
                self.params[key] = np.float(np.atleast_1d(buf['value'])[i])
            #::: if it's a "coupled parameter", then write the string of the key it is coupled to
            else:
                self.params[key] = np.atleast_1d(buf['value'])[i]
                
        #::: automatically set default params if they were not given
        self.params['automatically set:'] = ''                                 #just for printing
        for companion in self.settings['companions_all']:
            for inst in self.settings['inst_all']:
                
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: ellc defaults
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                if 'dil_'+inst not in self.params:
                    self.params['dil_'+inst] = 0.
                
                if companion+'_rr' not in self.params:
                    self.params[companion+'_rr'] = None
                    
                if companion+'_rsuma' not in self.params:
                    self.params[companion+'_rsuma'] = None
                    
                if companion+'_cosi' not in self.params:
                    self.params[companion+'_cosi'] = 0.
                    
                if companion+'_epoch' not in self.params:
                    self.params[companion+'_epoch'] = None
                    
                if companion+'_period' not in self.params:
                    self.params[companion+'_period'] = None
                    raise ValueError('It seems like you forgot to include the params for companion '+companion+' in the params.csv file...')
                    
                if companion+'_sbratio_'+inst not in self.params:
                    self.params[companion+'_sbratio_'+inst] = 0.               
                    
                if companion+'_a' not in self.params:
                    self.params[companion+'_a'] = None
                    
                if companion+'_q' not in self.params:
                    self.params[companion+'_q'] = 1.
                    
                if companion+'_K' not in self.params:
                    self.params[companion+'_K'] = 0.
                
                if companion+'_f_c' not in self.params:
                    self.params[companion+'_f_c'] = 0.
                    
                if companion+'_f_s' not in self.params:
                    self.params[companion+'_f_s'] = 0.
                    
                if 'host_ldc_'+inst not in self.params:
                    self.params['host_ldc_'+inst] = None
                    
                if companion+'_ldc_'+inst not in self.params:
                    self.params[companion+'_ldc_'+inst] = None
                    
                if 'host_gdc_'+inst not in self.params:
                    self.params['host_gdc_'+inst] = None
                    
                if companion+'_gdc_'+inst not in self.params:
                    self.params[companion+'_gdc_'+inst] = None
                    
                if 'didt_'+inst not in self.params:
                    self.params['didt_'+inst] = None
                    
                if 'domdt_'+inst not in self.params:
                    self.params['domdt_'+inst] = None
                    
                if 'host_rotfac_'+inst not in self.params:
                    self.params['host_rotfac_'+inst] = 1.
                    
                if companion+'_rotfac_'+inst not in self.params:
                    self.params[companion+'_rotfac_'+inst] = 1.
                    
                if 'host_hf_'+inst not in self.params:
                    self.params['host_hf_'+inst] = 1.5
                    
                if companion+'_hf_'+inst not in self.params:
                    self.params[companion+'_hf_'+inst] = 1.5
                    
                if 'host_bfac_'+inst not in self.params:
                    self.params['host_bfac_'+inst] = None
                    
                if companion+'_bfac_'+inst not in self.params:
                    self.params[companion+'_bfac_'+inst] = None
                    
                if 'host_geom_albedo_'+inst not in self.params:
                    self.params['host_geom_albedo_'+inst] = None
                    
                if companion+'_geom_albedo_'+inst not in self.params:
                    self.params[companion+'_geom_albedo_'+inst] = None
                    
                if 'host_lambda_'+inst not in self.params:
                    self.params['host_lambda_'+inst] = None
                    
                if companion+'_lambda_'+inst not in self.params:
                    self.params[companion+'_lambda_'+inst] = None
                    
                if 'host_vsini' not in self.params:
                    self.params['host_vsini'] = None
                    
                if companion+'_vsini' not in self.params:
                    self.params[companion+'_vsini'] = None
                    
                if 'host_spots_'+inst not in self.params:
                    self.params['host_spots_'+inst] = None
                    
                if companion+'_spots_'+inst not in self.params:
                    self.params[companion+'_spots_'+inst] = None
                    
                    
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: calculate number of spots
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                self.settings['host_N_spots_'+inst] = int( sum('host_spot_'+inst in s for s in self.params.keys())/4. )
#                self.settings[companion+'_N_spots_'+inst] = int( sum(companion+'_spots_'+inst in s for s in self.params.keys())/4. )
                    
                
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: ttvs
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                if 'ttv_'+inst not in self.params:
#                    self.params['ttv_'+inst] = 0.
                    
                    
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: to avoid a bug in ellc, if either property is >0, set the other to 1-15 (not 0):
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                if self.params[companion+'_geom_albedo_'+inst] is not None:
                    if (self.params[companion+'_sbratio_'+inst] == 0) and (self.params[companion+'_geom_albedo_'+inst] > 0):
                        self.params[companion+'_sbratio_'+inst] = 1e-15               #this is to avoid a bug in ellc
                    if (self.params[companion+'_sbratio_'+inst] > 0) and (self.params[companion+'_geom_albedo_'+inst] == 0):
                        self.params[companion+'_geom_albedo_'+inst] = 1e-15           #this is to avoid a bug in ellc
              
                
       
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: baseline_gp backwards compatability:
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_all']: 
            if inst in self.settings['inst_phot']: kkey='flux'
            elif inst in self.settings['inst_rv']: kkey='rv'
            if 'baseline_gp1_'+kkey+'_'+inst in self.params:
                self.params['baseline_gp_matern32_lnsigma_'+kkey+'_'+inst] = 1.*self.params['baseline_gp1_'+kkey+'_'+inst]
                warnings.warn('Deprecation warning. You are using outdated keywords. Automatically renaming '+'baseline_gp1_'+kkey+'_'+inst+' ---> '+'baseline_gp_matern32_lnsigma_'+kkey+'_'+inst)
            if 'baseline_gp2_'+kkey+'_'+inst in self.params:
                self.params['baseline_gp_matern32_lnrho_'+kkey+'_'+inst]   = 1.*self.params['baseline_gp2_'+kkey+'_'+inst]
                warnings.warn('Deprecation warning. You are using outdated keywords. Automatically renaming '+'baseline_gp2_'+kkey+'_'+inst+' ---> '+'baseline_gp_matern32_lnrho_'+kkey+'_'+inst)
        
        
        
        #::: coupled params
        if 'coupled_with' in buf.dtype.names:
            self.coupled_with = buf['coupled_with']
        else:
            self.coupled_with = [None]*len(self.allkeys)
            
            
            
        #::: deal with coupled params
        for i, key in enumerate(self.allkeys):
            if isinstance(self.coupled_with[i], str) and (len(self.coupled_with[i])>0):
                self.params[key] = self.params[self.coupled_with[i]]           #luser proof: automatically set the values of the params coupled to another param
                buf['fit'][i] = 0                                              #luser proof: automatically set fit=0 for the params coupled to another param
        
        
        
        #::: mark to be fitted params
        self.ind_fit = (buf['fit']==1)                  #len(all rows in params.csv)
        
        self.fitkeys = buf['name'][ self.ind_fit ]      #len(ndim)
        self.fitlabels = self.labels[ self.ind_fit ]    #len(ndim)
        self.fitunits = self.units[ self.ind_fit ]      #len(ndim)
        self.fittruths = self.truths[ self.ind_fit ]    #len(ndim)
        self.theta_0 = buf['value'][ self.ind_fit ]     #len(ndim)
        
        if 'init_err' in buf.dtype.names:
            self.init_err = buf['init_err'][ self.ind_fit ] #len(ndim)
        else:
            self.init_err = 1e-8
        
        self.bounds = [ str(item).split(' ') for item in buf['bounds'][ self.ind_fit ] ] #len(ndim)
        for i, item in enumerate(self.bounds):
            if item[0] in ['uniform', 'normal']:
                self.bounds[i] = [ item[0], np.float(item[1]), np.float(item[2]) ]
            elif item[0] in ['trunc_normal']:
                self.bounds[i] = [ item[0], np.float(item[1]), np.float(item[2]), np.float(item[3]), np.float(item[4]) ]
            else:
                raise ValueError('Bounds have to be "uniform", "normal" or "trunc_normal". Input from "params.csv" was "'+self.bounds[i][0]+'".')
    
        self.ndim = len(self.theta_0)                   #len(ndim)

    
    
        #::: check if all initial guess lie within their bounds
        for th, b, key in zip(self.theta_0, self.bounds, self.fitkeys):
            
            if (b[0] == 'uniform') and not (b[1] <= th <= b[2]): 
                raise ValueError('The initial guess for '+key+' lies outside of its bounds.')
                
            elif (b[0] == 'normal') and ( np.abs(th - b[1]) > 3*b[2] ):
                answer = input('The initial guess for '+key+' lies more than 3 sigma from its prior\n'+\
                      'What do you want to do?\n'+\
                      '1 : continue at any sacrifice \n'+\
                      '2 : stop and let me fix the params.csv file \n')
                if answer==1: 
                    pass
                else:
                    raise ValueError('User aborted the run.')
                    
            elif (b[0] == 'trunc_normal') and not (b[1] <= th <= b[2]): 
                raise ValueError('The initial guess for '+key+' lies outside of its bounds.')
                
            elif (b[0] == 'trunc_normal') and ( np.abs(th - b[3]) > 3*b[4] ): 
                answer = input('The initial guess for '+key+' lies more than 3 sigma from its prior\n'+\
                      'What do you want to do?\n'+\
                      '1 : continue at any sacrifice \n'+\
                      '2 : stop and let me fix the params.csv file \n')
                if answer==1: 
                    pass
                else:
                    raise ValueError('User aborted the run.')
            
            
    

    ###############################################################################
    #::: load data
    ###############################################################################
    def load_data(self):
        '''
        Example: 
        -------
            A lightcurve is stored as
                data['TESS']['time'], data['TESS']['flux']
            A RV curve is stored as
                data['HARPS']['time'], data['HARPS']['flux']
        '''
        self.fulldata = {}
        self.data = {}
        for inst in self.settings['inst_phot']:
            time, flux, flux_err = np.genfromtxt(os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:3]     
            if any(np.isnan(time)) or any(np.isnan(flux)) or any(np.isnan(flux_err)):
                raise ValueError('There are NaN values in "'+inst+'.csv". Please exclude these rows from the file and restart.')
            if not all(np.diff(time)>=0):
                raise ValueError('The time array in "'+inst+'.csv" is not sorted. Please make sure the file is not corrupted, then sort it by time and restart.')
            elif not all(np.diff(time)>0):
                warnings.warn('There are repeated time stamps in the time array in "'+inst+'.csv". Please make sure the file is not corrupted (e.g. insuffiecient precision in your time stamps).')
#                overwrite = str(input('There are repeated time stamps in the time array in "'+inst+'.csv". Please make sure the file is not corrupted (e.g. insuffiecient precision in your time stamps).'+\
#                                      'What do you want to do?\n'+\
#                                      '1 : continue and hope for the best; no risk, no fun; #yolo\n'+\
#                                      '2 : abort\n'))
#                if (overwrite == '1'):
#                    pass
#                else:
#                    raise ValueError('User aborted operation.')
                
            self.fulldata[inst] = {
                          'time':time,
                          'flux':flux,
                          'err_scales_flux':flux_err/np.nanmean(flux_err)
                         }
            if (self.settings['fast_fit']) and (len(self.settings['inst_phot'])>0): 
                time, flux, flux_err = self.reduce_phot_data(time, flux, flux_err, inst=inst)
            self.data[inst] = {
                          'time':time,
                          'flux':flux,
                          'err_scales_flux':flux_err/np.nanmean(flux_err)
                         }
            
        for inst in self.settings['inst_rv']:
            time, rv, rv_err = np.genfromtxt(os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)         
            if not all(np.diff(time)>0):
                raise ValueError('Your time array in "'+inst+'.csv" is not sorted. You will want to check that...')
            self.data[inst] = {
                          'time':time,
                          'rv':rv,
                          'white_noise_rv':rv_err
                         }
        
        #::: also save the combined time series
        #::: for cases where all instruments are treated together
        #::: e.g. for stellar variability GPs
        self.data['inst_phot'] = {'time':[],'flux':[],'flux_err':[],'inst':[]}
        for inst in self.settings['inst_phot']:
            self.data['inst_phot']['time'] += list(self.data[inst]['time'])
            self.data['inst_phot']['flux'] += list(self.data[inst]['flux'])
            self.data['inst_phot']['flux_err'] += [inst]*len(self.data[inst]['time']) #errors will be sampled/derived later
            self.data['inst_phot']['inst'] += [inst]*len(self.data[inst]['time'])
        ind_sort = np.argsort(self.data['inst_phot']['time'])
        self.data['inst_phot']['ind_sort'] = ind_sort
        self.data['inst_phot']['time'] = np.array(self.data['inst_phot']['time'])[ind_sort]
        self.data['inst_phot']['flux'] = np.array(self.data['inst_phot']['flux'])[ind_sort]
        self.data['inst_phot']['flux_err'] = np.array(self.data['inst_phot']['flux_err'])[ind_sort]      
        self.data['inst_phot']['inst'] = np.array(self.data['inst_phot']['inst'])[ind_sort]
    
        self.data['inst_rv'] = {'time':[],'rv':[],'rv_err':[],'inst':[]}
        for inst in self.settings['inst_rv']:
            self.data['inst_rv']['time'] += list(self.data[inst]['time'])
            self.data['inst_rv']['rv'] += list(self.data[inst]['rv'])
            self.data['inst_rv']['rv_err'] += list(np.nan*self.data[inst]['rv']) #errors will be sampled/derived later
            self.data['inst_rv']['inst'] += [inst]*len(self.data[inst]['time'])
        ind_sort = np.argsort(self.data['inst_rv']['time'])
        self.data['inst_rv']['ind_sort'] = ind_sort
        self.data['inst_rv']['time'] = np.array(self.data['inst_rv']['time'])[ind_sort]
        self.data['inst_rv']['rv'] = np.array(self.data['inst_rv']['rv'])[ind_sort]
        self.data['inst_rv']['rv_er'] = np.array(self.data['inst_rv']['rv_err'])[ind_sort]   
        self.data['inst_rv']['inst'] = np.array(self.data['inst_rv']['inst'])[ind_sort]

        
            
    ###############################################################################
    #::: change epoch
    ###############################################################################
    
    def my_truncnorm_isf(q,a,b,mean,std):
        a_scipy = 1.*(a - mean) / std
        b_scipy = 1.*(b - mean) / std
        return truncnorm.isf(q,a_scipy,b_scipy,loc=mean,scale=std)


    def change_epoch(self):
        '''
        change epoch entry from params.csv to set epoch into the middle of the range
        '''
        #::: for all companions
        for companion in self.settings['companions_all']:
            
            #::: get data time range
            alldata = []
            for inst in self.settings['inst_for_'+companion+'_epoch']:
                alldata += list(self.data[inst]['time'])
            start = np.nanmin( alldata )
            end = np.nanmax( alldata )
            
            #::: get the given values
            user_epoch  = 1.*self.params[companion+'_epoch']
            period      = 1.*self.params[companion+'_period']
#            buf = self.bounds[ind_e].copy()
                
            #::: calculate the true first_epoch
            if 'fast_fit_width' in self.settings and self.settings['fast_fit_width'] is not None:
                width = self.settings['fast_fit_width']
            else:
                width = 0
            first_epoch = get_first_epoch(alldata, self.params[companion+'_epoch'], self.params[companion+'_period'], width=width)
            
            #::: calculate the mid_epoch (in the middle of the data set)
            N = int(np.round((end-start)/2./period))
            self.settings['mid_epoch'] = first_epoch + N * period
            
            #::: calculate how much the user_epoch has to be shifted to get the mid_epoch
            N_shift = int(np.round((self.settings['mid_epoch']-user_epoch)/period))
            
            #::: set the new initial guess (and truth)
            self.params[companion+'_epoch'] = 1.*self.settings['mid_epoch']
           
            #::: also shift the truth (implies that the turth epoch is set where the initial guess is)
            if (self.fittruths is not None) and (companion+'_epoch' in self.fitkeys):
                ind_e = np.where(self.fitkeys==companion+'_epoch')[0][0]
                ind_p = np.where(self.fitkeys==companion+'_period')[0][0]
                N_truth_shift = int(np.round((self.settings['mid_epoch']-self.fittruths[ind_e])/self.fittruths[ind_p]))
                self.fittruths[ind_e] += N_truth_shift * self.fittruths[ind_p]
                
            #::: if a fit param, also update the bounds accordingly
            if (N_shift != 0) and (companion+'_epoch' in self.fitkeys):
                ind_e = np.where(self.fitkeys==companion+'_epoch')[0][0]
                ind_p = np.where(self.fitkeys==companion+'_period')[0][0]
                
#                print('\n')
#                print('############################################################################')
#                print('user_epoch', user_epoch, self.bounds[ind_e])
#                print('user_period', period, self.bounds[ind_p])
#                print('----------------------------------------------------------------------------')
                  
                #::: set the new initial guess
                self.theta_0[ind_e] = 1.*self.settings['mid_epoch']
                
                #::: get the bounds / errors
                #::: if the epoch and period priors are both uniform
                if (self.bounds[ind_e][0] == 'uniform') & (self.bounds[ind_p][0] == 'uniform'):
                    if N_shift > 0:
                        self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][1] #lower bound
                        self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][2] #upper bound
                    elif N_shift < 0:
                        self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][2] #lower bound; period bounds switched if N_shift is negative
                        self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][1] #upper bound; period bounds switched if N_shift is negative
                
                #::: if the epoch and period priors are both normal
                elif (self.bounds[ind_e][0] == 'normal') & (self.bounds[ind_p][0] == 'normal'):
                    self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][1] #mean (in case the prior-mean is not the initial-guess-mean)
                    self.bounds[ind_e][2] = np.sqrt( self.bounds[ind_e][2]**2 + N_shift**2 * self.bounds[ind_p][2]**2 ) #std (in case the prior-mean is not the initial-guess-mean)
                                        
                #::: if the epoch and period priors are both trunc_normal
                elif (self.bounds[ind_e][0] == 'trunc_normal') & (self.bounds[ind_p][0] == 'trunc_normal'):
                    if N_shift > 0:
                        self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][1] #lower bound
                        self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][2] #upper bound
                    elif N_shift < 0:
                        self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][2] #lower bound; period bounds switched if N_shift is negative
                        self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][1] #upper bound; period bounds switched if N_shift is negative
                    self.bounds[ind_e][3] = self.bounds[ind_e][3] + N_shift * self.bounds[ind_p][3] #mean (in case the prior-mean is not the initial-guess-mean)
                    self.bounds[ind_e][4] = np.sqrt( self.bounds[ind_e][4]**2 + N_shift**2 * self.bounds[ind_p][4]**2 ) #std (in case the prior-mean is not the initial-guess-mean)
            
                #::: if the epoch prior is uniform and period prior is normal
                elif (self.bounds[ind_e][0] == 'uniform') & (self.bounds[ind_p][0] == 'normal'):
                    self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * (period + self.bounds[ind_p][2]) #lower bound epoch + Nshift * period + Nshift * std_period
                    self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * (period + self.bounds[ind_p][2]) #upper bound + Nshift * period + Nshift * std_period

                #::: if the epoch prior is uniform and period prior is trunc_normal
                elif (self.bounds[ind_e][0] == 'uniform') & (self.bounds[ind_p][0] == 'trunc_normal'):
                    self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * (period + self.bounds[ind_p][4]) #lower bound epoch + Nshift * period + Nshift * std_period
                    self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * (period + self.bounds[ind_p][4]) #upper bound + Nshift * period + Nshift * std_period

                elif (self.bounds[ind_e][0] == 'normal') & (self.bounds[ind_p][0] == 'uniform'):
                    raise ValueError('shift_epoch with different priors for epoch and period is not yet implemented.')
                    
                elif (self.bounds[ind_e][0] == 'normal') & (self.bounds[ind_p][0] == 'trunc_normal'):
                    raise ValueError('shift_epoch with different priors for epoch and period is not yet implemented.')
                    
                elif (self.bounds[ind_e][0] == 'trunc_normal') & (self.bounds[ind_p][0] == 'uniform'):
                    raise ValueError('shift_epoch with different priors for epoch and period is not yet implemented.')
                    
                elif (self.bounds[ind_e][0] == 'trunc_normal') & (self.bounds[ind_p][0] == 'normal'):
                    raise ValueError('shift_epoch with different priors for epoch and period is not yet implemented.')
                    
                else:
                    raise ValueError('Parameters "bounds" have to be "uniform", "normal" or "trunc_normal".')
                    
        
#                print('first_epoch; N', first_epoch, N)
#                print('mid_epoch, error; N_shift', self.settings['mid_epoch'], N_shift)
#                print('----------------------------------------------------------------------------')
#                print('new epoch:', self.settings['mid_epoch'], self.bounds[ind_e])
#                print('############################################################################')
#                print('\n')
#                err
                
#                print('\n', 'New epochs:')
#                print(self.params[companion+'_epoch'])
#                    
        '''
        #::: change epoch entry from params.csv to set epoch into the middle of the range
        for companion in self.settings['companions_all']:
            #::: get data time range
            alldata = []
            for inst in self.settings['inst_for_'+companion+'_epoch']:
                alldata += list(self.data[inst]['time'])
            start = np.nanmin( alldata )
            end = np.nanmax( alldata )
            
#            import matplotlib.pyplot as plt
#            plt.figure()
#            plt.plot(all_data, np.ones_like(all_data), 'bo')
            
            first_epoch = 1.*self.params[companion+'_epoch']
            period      = 1.*self.params[companion+'_period']
            
#            plt.axvline(first_epoch, color='r', lw=2)
            
            #::: place the first_epoch at the start of the data to avoid luser mistakes
#            if start<=first_epoch:
#                first_epoch -= int(np.round((first_epoch-start)/period)) * period
#            else:
#                first_epoch += int(np.round((start-first_epoch)/period)) * period
            if 'fast_fit_width' in self.settings and self.settings['fast_fit_width'] is not None:
                width = self.settings['fast_fit_width']
            else:
                width = 0
            first_epoch = get_first_epoch(alldata, self.params[companion+'_epoch'], self.params[companion+'_period'], width=width)
            
#            plt.axvline(first_epoch, color='b', lw=2)
                
            #::: place epoch_for_fit into the middle of all data
            N = int(np.round((end-start)/2./period))
            epoch_shift = N * period 
            epoch_for_fit = first_epoch + epoch_shift
            
            #::: update params
            self.params[companion+'_epoch'] = 1.*epoch_for_fit
            
            #::: update theta, bounds and fittruths
            ind_epoch_fitkeys = np.where(self.fitkeys==companion+'_epoch')[0]
            if len(ind_epoch_fitkeys):
                ind_epoch_fitkeys = ind_epoch_fitkeys[0]
                buf = 1.*self.theta_0[ind_epoch_fitkeys]
                self.theta_0[ind_epoch_fitkeys]    = 1.*epoch_for_fit                #initial guess
                
                #::: update fittruhts
                self.fittruths[ind_epoch_fitkeys] += epoch_shift 
                
                #:::change bounds if uniform bounds
                if self.bounds[ind_epoch_fitkeys][0] == 'uniform':
                    lower = buf - self.bounds[ind_epoch_fitkeys][1]
                    upper = self.bounds[ind_epoch_fitkeys][2] - buf
                    self.bounds[ind_epoch_fitkeys][1]  = epoch_for_fit - lower           #lower bound
                    self.bounds[ind_epoch_fitkeys][2]  = epoch_for_fit + upper           #upper bound
                
                #:::change bounds if normal or trunc_normal bounds
                elif self.bounds[ind_epoch_fitkeys][0] == 'normal':
                    mean = 1.*self.theta_0[ind_epoch_fitkeys]
                    std = 1.*self.bounds[ind_epoch_fitkeys][2]
                    self.bounds[ind_epoch_fitkeys][1]  = mean         
                    self.bounds[ind_epoch_fitkeys][2]  = std        
                    
                elif self.bounds[ind_epoch_fitkeys][0] == 'trunc_normal':
                    lower = buf - self.bounds[ind_epoch_fitkeys][1]
                    upper = self.bounds[ind_epoch_fitkeys][2] - buf
                    mean = 1.*self.theta_0[ind_epoch_fitkeys]
                    std = 1.*self.bounds[ind_epoch_fitkeys][4]
                    self.bounds[ind_epoch_fitkeys][1]  = epoch_for_fit - lower           #lower bound
                    self.bounds[ind_epoch_fitkeys][2]  = epoch_for_fit + upper
                    self.bounds[ind_epoch_fitkeys][3]  = mean         
                    self.bounds[ind_epoch_fitkeys][4]  = std        
                    
                else:
                    raise ValueError('Parameters "bounds" have to be "uniform", "normal" or "trunc_normal".')
        '''


    ###############################################################################
    #::: reduce_phot_data
    ###############################################################################
    def reduce_phot_data(self, time, flux, flux_err, inst=None):
        ind_in = []
              
        for companion in self.settings['companions_phot']:
            epoch  = self.params[companion+'_epoch']
            period = self.params[companion+'_period']
            width  = self.settings['fast_fit_width']
            if self.settings['secondary_eclipse']:
                ind_ecl1x, ind_ecl2x, ind_outx = index_eclipses(time,epoch,period,width,width) #TODO: currently this assumes width_occ == width_tra
                ind_in += list(ind_ecl1x)
                ind_in += list(ind_ecl2x)
                self.fulldata[inst][companion+'_ind_ecl1'] = ind_ecl1x
                self.fulldata[inst][companion+'_ind_ecl2'] = ind_ecl2x
                self.fulldata[inst][companion+'_ind_out'] = ind_outx
            else:
                ind_inx, ind_outx = index_transits(time,epoch,period,width)
                ind_in += list(ind_inx)
                self.fulldata[inst][companion+'_ind_in'] = ind_inx
                self.fulldata[inst][companion+'_ind_out'] = ind_outx
                
        ind_in = np.sort(np.unique(ind_in))
        self.fulldata[inst]['all_ind_in'] = ind_in
        self.fulldata[inst]['all_ind_out'] = np.delete( np.arange(len(self.fulldata[inst]['time'])), ind_in )
        
        if len(ind_in)==0:
            raise ValueError(inst+'.csv does not contain any in-transit data. Check that your epoch and period guess are correct.')
        
        time = time[ind_in]
        flux = flux[ind_in]
        flux_err = flux_err[ind_in]
        return time, flux, flux_err
    
    
    
    
    ###############################################################################
    #::: prepare TTV fit (if chosen)
    ###############################################################################
    def prepare_ttv_fit(self):
        '''
        this must be run *after* reduce_phot_data()
        '''
        
        for companion in self.settings['companions_phot']:
            all_times = []
            all_flux = []
            for inst in self.settings['inst_phot']:
                all_times += list(self.data[inst]['time'])
                all_flux += list(self.data[inst]['flux'])
            
            self.data[companion+'_tmid_observed_transits'] = get_tmid_observed_transits(all_times,self.params[companion+'_epoch'],self.params[companion+'_period'],self.settings['fast_fit_width'])
        
            #::: plots
            # if self.settings['fit_ttvs']:  
            #     flux_min = np.nanmin(all_flux)
            #     flux_max = np.nanmax(all_flux)
            #     N_days = int( np.max(all_times) - np.min(all_times) )
            #     figsizex = np.min( [1, int(N_days/20.)] )*5
            #     fig, ax = plt.subplots(figsize=(figsizex, 4)) #figsize * 5 for every 20 days
            #     for inst in self.settings['inst_phot']:
            #         ax.plot(self.data[inst]['time'], self.data[inst]['flux'],ls='none',marker='.',label=inst)
            #     ax.plot( self.data[companion+'_tmid_observed_transits'], np.ones_like(self.data[companion+'_tmid_observed_transits'])*0.995*flux_min, 'k^' )
            #     for i, tmid in enumerate(self.data[companion+'_tmid_observed_transits']):
            #         ax.text( tmid, 0.9925*flux_min, str(i+1), ha='center' )  
            #     ax.set(ylim=[0.99*flux_min, flux_max], xlabel='Time (BJD)', ylabel='Realtive Flux') 
            #     if not os.path.exists( os.path.join(self.datadir,'results') ):
            #         os.makedirs(os.path.join(self.datadir,'results'))
            #     ax.legend()
            #     fname = os.path.join(self.datadir,'results','preparation_for_TTV_fit_'+companion+'.pdf')
            #     if os.path.exists(fname):
            #         overwrite = str(input('Figure "preparation_for_TTV_fit_'+companion+'.pdf" already exists.\n'+\
            #                               'What do you want to do?\n'+\
            #                               '1 : overwrite it\n'+\
            #                               '2 : skip it and move on\n'))
            #         if (overwrite == '1'):
            #             fig.savefig(fname, bbox_inches='tight' )    
            #         else:
            #             pass        
            #     plt.close(fig)
            
            width = self.settings['fast_fit_width']
            for inst in self.settings['inst_phot']:
                time = self.data[inst]['time']
                for i, t in enumerate(self.data[companion+'_tmid_observed_transits']):
                    ind = np.where((time >= (t - width/2.)) & (time <= (t + width/2.)))[0]
                    self.data[inst][companion+'_ind_time_transit_'+str(i+1)] = ind
                    self.data[inst][companion+'_time_transit_'+str(i+1)] = time[ind]
            
                
                
            
    ###############################################################################
    #::: stellar priors
    ###############################################################################
    def load_stellar_priors(self, N_samples=10000):
        if os.path.exists(os.path.join(self.datadir,'params_star.csv')) and (self.settings['use_host_density_prior'] is True):
            buf = np.genfromtxt( os.path.join(self.datadir,'params_star.csv'), delimiter=',', names=True, dtype=None, encoding='utf-8', comments='#' )
            radius = simulate_PDF(buf['R_star'], buf['R_star_lerr'], buf['R_star_uerr'], size=N_samples, plot=False) * 6.957e10 #in cgs
            mass = simulate_PDF(buf['M_star'], buf['M_star_lerr'], buf['M_star_uerr'], size=N_samples, plot=False) * 1.9884754153381438e+33 #in cgs
            volume = (4./3.)*np.pi*radius**3 #in cgs
            density = mass / volume #in cgs
            self.params_star = {'R_star_median':buf['R_star'],
                                'R_star_lerr':buf['R_star_lerr'],
                                'R_star_uerr':buf['R_star_uerr'],
                                'M_star_median':buf['M_star'],
                                'M_star_lerr':buf['M_star_lerr'],
                                'M_star_uerr':buf['M_star_uerr']
                                }
            self.external_priors['host_density'] = ['normal', np.median(density), np.max( [np.median(density)-np.percentile(density,16), np.percentile(density,84)-np.median(density)] ) ] #in cgs
            
            