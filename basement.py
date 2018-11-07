#!/usr/bin/env python2
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
import os, sys
import collections
from datetime import datetime
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 

#::: my modules
from exoworlds.lightcurves import index_transits, index_eclipses

                     
    
    
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
        All the variables needed for allesfitter.MCMC_fit
        '''
        
        self.now = datetime.now().isoformat()
        
        self.datadir = datadir
        
        self.load_settings()
        self.load_params()
        self.load_data()
        self.change_epoch()
        
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
            if (self.settings['baseline_'+key+'_'+inst] == 'sample_GP') &\
               (self.settings['error_'+key+'_'+inst] != 'sample'):
                   raise ValueError('If you want to use sample_GP, you will want to sample the jitters, too!')
            
                     
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
        planets_phot 
        planets_rv
        inst_phot
        inst_rv
        ###############################################################################
        # Fit performance settings
        ###############################################################################
        fast_fit                              : optional. Default is False.
        fast_fit_width                        : optional. Default is 8./24.
        secondary_eclipse            : optional. Default is False.
        multiprocess                          : optional. Default is False.
        multiprocess_cores                    : optional. Default is cpu_count()-1.
        ###############################################################################
        # MCMC settings
        ###############################################################################
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
            
        rows = np.genfromtxt( os.path.join(self.datadir,'settings.csv'), dtype=None, delimiter=',' )

#        self.settings = {r[0]:r[1] for r in rows}
        self.settings = collections.OrderedDict( [('user-given:','')]+[ (r[0],r[1]) for r in rows ]+[('automatically set:','')] )
        
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: General settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for key in ['planets_phot', 'planets_rv', 'inst_phot', 'inst_rv']:
            if key not in self.settings:
                self.settings[key] = []
            elif len(self.settings[key]): 
                self.settings[key] = str(self.settings[key]).split(' ')
            else:                       
                self.settings[key] = []
        
        self.settings['planets_all']  = list(np.unique(self.settings['planets_phot']+self.settings['planets_rv'])) #sorted by b, c, d...
        self.settings['inst_all'] = list(unique( self.settings['inst_phot']+self.settings['inst_rv'] )) #sorted like user input
    
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Epoch settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for planet in self.settings['planets_all']:
            if 'inst_for_'+planet+'_epoch' not in self.settings:
                self.settings['inst_for_'+planet+'_epoch'] = 'all'
        
            if self.settings['inst_for_'+planet+'_epoch'] in ['all','none']:
                self.settings['inst_for_'+planet+'_epoch'] = self.settings['inst_all']
            else:
                if len(self.settings['inst_for_'+planet+'_epoch']): 
                    self.settings['inst_for_'+planet+'_epoch'] = str(self.settings['inst_for_'+planet+'_epoch']).split(' ')
                else:                       
                    self.settings['inst_for_'+planet+'_epoch'] = []
        
        
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
        #::: Fast fit
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        self.settings['fast_fit'] = set_bool(self.settings['fast_fit'])
        
        if ('fast_fit_width' in self.settings.keys()) and len(self.settings['fast_fit_width']):
            self.settings['fast_fit_width'] = np.float(self.settings['fast_fit_width'])
        else:
            self.settings['fast_fit_width'] = 8./24.
                
        
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
        if 'mcmc_nwalkers' not in self.settings: 
            self.settings['mcmc_nwalkers'] = 100
        if 'mcmc_total_steps' not in self.settings: 
            self.settings['mcmc_total_steps'] = 2000
        if 'mcmc_burn_steps' not in self.settings: 
            self.settings['mcmc_burn_steps'] = 1000
        if 'mcmc_thin_by' not in self.settings: 
            self.settings['mcmc_thin_by'] = 1
                
        for key in ['mcmc_nwalkers','mcmc_total_steps','mcmc_burn_steps','mcmc_thin_by']:
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
        #::: Limb darkening laws
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_phot']:
            if 'ld_law_'+inst not in self.settings: 
                self.settings['ld_law_'+inst] = 'quad'
                
                
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Baselines
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_phot']:
            for key in ['flux']:
                if 'baseline_'+key+'_'+inst not in self.settings: 
                    self.settings['baseline_'+key+'_'+inst] = 'hybrid_spline'

        for inst in self.settings['inst_rv']:
            for key in ['rv']:
                if 'baseline_'+key+'_'+inst not in self.settings: 
                    self.settings['baseline_'+key+'_'+inst] = 'hybrid_offset'
                    
                
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
                
        
                
    ###############################################################################
    #::: load params
    ###############################################################################
    def load_params(self):
        '''
        #name	value	fit	bounds	label	unit
        #b_: planet name; _key : flux/rv/centd; _inst : instrument name					
        #dilution per instrument					
        light_3_TESS	0	0	none	$D_\mathrm{TESS}$	
        light_3_HATS	0.14	1	trunc_normal 0 1 0.14 0.1	$D_\mathrm{HATS}$	
        light_3_FTS_i	0	0	none	$D_\mathrm{FTS_i}$	
        light_3_GROND_g	0	0	none	$D_\mathrm{GROND_g}$	
        light_3_GROND_r	0	0	none	$D_\mathrm{GROND_r}$	
        light_3_GROND_i	0	0	none	$D_\mathrm{GROND_i}$	
        light_3_GROND_z	0	0	none	$D_\mathrm{GROND_i}$	
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
        #brightness per instrument per planet					
        b_sbratio_TESS	0	0	none	$J_{b;\mathrm{TESS}}$	
        b_sbratio_HATS	0	0	none	$J_{b;\mathrm{HATS}}$	
        b_sbratio_FTS_i	0	0	none	$J_{b;\mathrm{FTS_i}}$	
        b_sbratio_GROND_g	0	0	none	$J_{b;\mathrm{GROND_g}}$	
        b_sbratio_GROND_r	0	0	none	$J_{b;\mathrm{GROND_r}}$	
        b_sbratio_GROND_i	0	0	none	$J_{b;\mathrm{GROND_i}}$	
        b_sbratio_GROND_z	0	0	none	$J_{b;\mathrm{GROND_z}}$	
        #planet b astrophysical params					
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
        b_ttv_TESS	0	0	trunc_normal -0.04 0.04 0 0.0007	$TTV_\mathrm{TESS}$	$\mathrm{d}$
        b_ttv_HATS	0	0	trunc_normal -0.04 0.04 0 0.0007	$TTV_\mathrm{HATS}$	$\mathrm{d}$
        b_ttv_FTS_i	0	0	trunc_normal -0.04 0.04 0 0.0007	$TTV_\mathrm{FTS_i}$	$\mathrm{d}$
        b_ttv_GROND_g	0	0	trunc_normal -0.04 0.04 0 0.0007	$TTV_\mathrm{GROND_g}$	$\mathrm{d}$
        b_ttv_GROND_r	0	0	trunc_normal -0.04 0.04 0 0.0007	$TTV_\mathrm{GROND_r}$	$\mathrm{d}$
        b_ttv_GROND_i	0	0	trunc_normal -0.04 0.04 0 0.0007	$TTV_\mathrm{GROND_i}$	$\mathrm{d}$
        b_ttv_GROND_z	0	0	trunc_normal -0.04 0.04 0 0.0007	$TTV_\mathrm{GROND_i}$	$\mathrm{d}$
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
        
        
    
        buf = np.genfromtxt(os.path.join(self.datadir,'params.csv'), delimiter=',',comments='#',dtype=None,names=True)
        
        self.allkeys = buf['name'] #len(all rows in params.csv)
        self.labels = buf['label'] #len(all rows in params.csv)
        self.units = buf['unit']   #len(all rows in params.csv)
        
        if 'truth' in buf.dtype.names:
            self.truths = buf['truth'] #len(all rows in params.csv)
        else:
            self.truths = np.nan * np.ones(len(self.allkeys))
            
        self.params = collections.OrderedDict()           #len(all rows in params.csv)
        self.params['user-given:'] = ''
        for i,key in enumerate(self.allkeys):
            self.params[key] = np.float(buf['value'][i])
         
        #::: automatically set default params if they were not given
        self.params['automatically set:'] = ''
        for planet in self.settings['planets_phot']:
            for inst in self.settings['inst_phot']:
                if 'light_3_'+inst not in self.params:
                    self.params['light_3_'+inst] = 0.
                if planet+'_sbratio_'+inst not in self.params:
                    self.params[planet+'_sbratio_'+inst] = 1e-12 #this is to avoid a bug in ellc
                if planet+'_geom_albedo_'+inst not in self.params:
                    self.params[planet+'_geom_albedo_'+inst] = 1e-12 #this is to avoid a bug in ellc
                    
        for planet in self.settings['planets_rv']:
            if planet+'_q' not in self.params:
                self.params[planet+'_q'] = 1.
            if planet+'_K' not in self.params:
                self.params[planet+'_K'] = 0.
                
        for planet in self.settings['planets_all']:
            if planet+'_f_c' not in self.params:
                self.params[planet+'_f_c'] = 0.
            if planet+'_f_s' not in self.params:
                self.params[planet+'_f_s'] = 0.
                
            
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
                raise ValueError('Bounds have to be "uniform" or "normal". Input from "params.csv" was "'+self.bounds[i][0]+'".')
    
        self.ndim = len(self.theta_0)                   #len(ndim)

    
    

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
        self.data = {}
        for inst in self.settings['inst_phot']:
            time, flux, flux_err = np.genfromtxt(os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)         
            if not all(np.diff(time)>0):
                raise ValueError('Your time array in "'+inst+'.csv" is not sorted. You will want to check that...')
            if self.settings['fast_fit']: 
                time, flux, flux_err = self.reduce_phot_data(time, flux, flux_err)
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
            
            
            
    ###############################################################################
    #::: change epoch
    ###############################################################################
    def change_epoch(self):
        
        #::: change epoch entry from params.csv to set epoch into the middle of the range
        for planet in self.settings['planets_all']:
            #::: get data time range
            all_data = []
            for inst in self.settings['inst_for_'+planet+'_epoch']:
                all_data += list(self.data[inst]['time'])
            start = np.nanmin( all_data )
            end = np.nanmax( all_data )
            
#            import matplotlib.pyplot as plt
#            plt.figure()
#            plt.plot(all_data, np.ones_like(all_data), 'bo')
            
            first_epoch = 1.*self.params[planet+'_epoch']
            period      = 1.*self.params[planet+'_period']
            
#            plt.axvline(first_epoch, color='r', lw=2)
            
            #::: place the first_epoch at the start of the data to avoid luser mistakes
            if start<=first_epoch:
                first_epoch -= int(np.round((first_epoch-start)/period)) * period
            else:
                first_epoch += int(np.round((start-first_epoch)/period)) * period
                
#            plt.axvline(first_epoch, color='b', lw=2)
                
            #::: place epoch_for_fit into the middle of all data
            epoch_shift = int(np.round((end-start)/2./period)) * period 
            epoch_for_fit = first_epoch + epoch_shift
            
            #::: update params
            self.params[planet+'_epoch'] = 1.*epoch_for_fit
            
            #::: update theta, bounds and fittruths
            ind_epoch_fitkeys = np.where(self.fitkeys==planet+'_epoch')[0]
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
           
            #::: print output (for testing only)
#            print('\nSetting epoch for planet '+planet)
#            print('\tfirst epoch, from params.csv file:', first_epoch)
#            
#            print('\nOrbital cycles since then:', int( (end-start) / period))
#            
#            print('\tepoch for fit, placed in the middle of the data range:', epoch_for_fit)
#            print('Theta for fit:', self.theta_0[ind_epoch_fitkeys])
#            print('Bounds[1] for fit:', self.bounds[ind_epoch_fitkeys][1])
#            print('Bounds[2] for fit:', self.bounds[ind_epoch_fitkeys][2])
##            
#            plt.axvline(epoch_for_fit, color='g', lw=2)
#            plt.axvspan(self.bounds[ind_epoch_fitkeys][1], self.bounds[ind_epoch_fitkeys][2], alpha=0.8, color='g')
#            plt.xlim([start-10,end+10])
#            plt.show()


    ###############################################################################
    #::: reduce_phot_data
    ###############################################################################
    def reduce_phot_data(self, time, flux, flux_err):
        ind_in = []
              
        for planet in self.settings['planets_phot']:
            epoch  = self.params[planet+'_epoch']
            period = self.params[planet+'_period']
            width  = self.settings['fast_fit_width']
            if self.settings['secondary_eclipse']:
                ind_ecl1, ind_ecl2, _ = index_eclipses(time,epoch,period,width,width) #TODO: currently this assumes width_occ == width_tra
                ind_in += list(ind_ecl1)
                ind_in += list(ind_ecl2)
            else:
                ind_in += list(index_transits(time,epoch,period,width)[0])
        ind_in = np.sort(ind_in)
        time = time[ind_in]
        flux = flux[ind_in]
        flux_err = flux_err[ind_in]
        return time, flux, flux_err
            