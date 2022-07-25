#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:17:06 2018

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
import os
import sys
import fnmatch
import collections
from datetime import datetime
from multiprocessing import cpu_count
import warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f'\n! WARNING:\n {msg}\ntype: {args[0]}, file: {args[1]}, line: {args[2]}\n'
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 
from scipy.stats import truncnorm

#::: allesfitter modules
from .exoworlds_rdx.lightcurves.index_transits import index_transits, index_eclipses, get_first_epoch, get_tmid_observed_transits
from .priors.simulate_PDF import simulate_PDF
from .utils.mcmc_move_translator import translate_str_to_move

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})
                     
    
    
    
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
    def __init__(self, datadir, quiet=False):
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
        
        print('Filling the Basement')
        
        self.quiet = quiet
        self.now = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        self.datadir = datadir
        self.outdir = os.path.join(datadir,'results') 
        if not os.path.exists( self.outdir ): os.makedirs( self.outdir )
        
        print('')
        self.logprint('\nallesfitter version')
        self.logprint('---------------------')
        self.logprint('v1.2.10')
        
        self.load_settings()
        self.load_params()
        self.load_data()
        
        if self.settings['shift_epoch']:
            try:
                self.change_epoch()
            except:
                warnings.warn('\nCould not shift epoch (you can peacefully ignore this warning if no period was given)\n')
        
        if self.settings['fit_ttvs']:  
            self.setup_ttv_fit()
        
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
        if not self.quiet:
            print(*text)
            original = sys.stdout
            with open( os.path.join(self.outdir,'logfile_'+self.now+'.log'), 'a' ) as f:
                sys.stdout = f
                print(*text)
            sys.stdout = original
        else:
            pass
        
        

    ###############################################################################
    #::: load settings
    ###############################################################################
    def load_settings(self):
        '''
        For the full list of options see www.allesfitter.com
        '''
        
        
        def set_bool(text):
            if text.lower() in ['true', '1']:
                return True
            else:
                return False
            
        
        def is_empty_or_none(key):
            return (key not in self.settings) or (str(self.settings[key]).lower() == 'none') or (len(self.settings[key])==0)
            
        
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
                warnings.warn('You are using outdated keywords. Automatically renaming '+name+' ---> '+rows[i][0]+'. Please fix this before the Duolingo owl comes to get you.') #, category=DeprecationWarning)
            if name[:6]=='ld_law':
                rows[i][0] = 'host_ld_law'+name[6:]
                warnings.warn('You are using outdated keywords. Automatically renaming '+name+' ---> '+rows[i][0]+'. Please fix this before the Duolingo owl comes to get you.') #, category=DeprecationWarning)
                
#        self.settings = {r[0]:r[1] for r in rows}
        self.settings = collections.OrderedDict( [('user-given:','')]+[ (r[0],r[1] ) for r in rows ]+[('automatically set:','')] )

        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Main settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'time_format' not in self.settings:
            self.settings['time_format'] = 'BJD_TDB'
            
        for key in ['companions_phot', 'companions_rv', 'inst_phot', 'inst_rv', 'inst_rv2']:
            if key not in self.settings:
                self.settings[key] = []
            elif len(self.settings[key]): 
                self.settings[key] = str(self.settings[key]).split(' ')
            else:                       
                self.settings[key] = []
        
        self.settings['companions_all']  = list(np.unique(self.settings['companions_phot']+self.settings['companions_rv'])) #sorted by b, c, d...
        self.settings['inst_all'] = list(unique( self.settings['inst_phot']+self.settings['inst_rv']+self.settings['inst_rv2'] )) #sorted like user input
    
        if len(self.settings['inst_phot'])==0 and len(self.settings['companions_phot'])>0:
            raise ValueError('No photometric instrument is selected, but photometric companions are given.')
        if len(self.settings['inst_rv'])==0 and len(self.settings['companions_rv'])>0:
           raise ValueError('No RV instrument is selected, but RV companions are given.')
            
            
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
            self.settings['shift_epoch'] = True 
            
            
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
        
        from pprint import pprint
        pprint(self.settings)
        
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


        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Phase variations
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('phase_variations' in self.settings.keys()) and len(self.settings['phase_variations']):
            warnings.warn('You are using outdated keywords. Automatically renaming "phase_variations" ---> "phase_curve".'+'. Please fix this before the Duolingo owl comes to get you.')
            self.settings['phase_curve'] = self.settings['phase_variations']
            
        if ('phase_curve' in self.settings.keys()) and len(self.settings['phase_curve']):
            self.settings['phase_curve'] = set_bool(self.settings['phase_curve'])
            if self.settings['phase_curve']==True:                
                # self.logprint('The user set phase_curve==True. Automatically set fast_fit=False and secondary_eclispe=True, and overwrite other settings.')
                self.settings['fast_fit'] = 'False'
                self.settings['secondary_eclipse'] = 'True'
        else:
            self.settings['phase_curve'] = False
            
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Fast fit
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('fast_fit' in self.settings.keys()) and len(self.settings['fast_fit']):
            self.settings['fast_fit'] = set_bool(self.settings['fast_fit'])
        else:
            self.settings['fast_fit'] = False
        
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
        #::: Host stellar density prior
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'use_tidal_eccentricity_prior' in self.settings:
            self.settings['use_tidal_eccentricity_prior'] = set_bool(self.settings['use_tidal_eccentricity_prior'] )
        else:
            self.settings['use_tidal_eccentricity_prior'] = False
            
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: TTVs
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('fit_ttvs' in self.settings.keys()) and len(self.settings['fit_ttvs']):
            self.settings['fit_ttvs'] = set_bool(self.settings['fit_ttvs'])
            if (self.settings['fit_ttvs']==True) and (self.settings['fast_fit']==False):
                raise ValueError('fit_ttvs==True, but fast_fit==False.'+\
                                 'Currently, you can only fit for TTVs if fast_fit==True.'+\
                                 'Please choose different settings.')
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
        if 'mcmc_moves' not in self.settings: 
            self.settings['mcmc_moves'] = 'DEMove'
            
        #::: make sure these are integers
        for key in ['mcmc_nwalkers','mcmc_pre_run_loops','mcmc_pre_run_steps',
                    'mcmc_total_steps','mcmc_burn_steps','mcmc_thin_by']:
            self.settings[key] = int(self.settings[key])
            
        #::: luser proof
        if self.settings['mcmc_total_steps'] <= self.settings['mcmc_burn_steps']:
            raise ValueError('Your setting for mcmc_total_steps must be larger than mcmc_burn_steps (check your settings.csv).')
                
            
        #::: translate the mcmc_move string into a list of emcee commands
        self.settings['mcmc_moves'] = translate_str_to_move(self.settings['mcmc_moves'])
            
        # N_evaluation_samples = int( 1. * self.settings['mcmc_nwalkers'] * (self.settings['mcmc_total_steps']-self.settings['mcmc_burn_steps']) / self.settings['mcmc_thin_by'] )
        # self.logprint('\nAnticipating ' + str(N_evaluation_samples) + 'MCMC evaluation samples.\n')
        # if N_evaluation_samples>200000:
        #     answer = input('It seems like you are asking for ' + str(N_evaluation_samples) + 'MCMC evaluation samples (calculated as mcmc_nwalkers * (mcmc_total_steps-mcmc_burn_steps) / mcmc_thin_by).'+\
        #                    'That is an aweful lot of samples.'+\
        #                    'What do you want to do?\n'+\
        #                    '1 : continue at any sacrifice\n'+\
        #                    '2 : abort and increase the mcmc_thin_by parameter in settings.csv (do not do this if you continued an old run!)\n')
        #     if answer==1: 
        #         pass
        #     else:
        #         raise ValueError('User aborted the run.')

        
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
        #::: host & companion grids, limb darkening laws, shapes, etc.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for companion in self.settings['companions_all']:
            for inst in self.settings['inst_all']:
                
                if 'host_grid_'+inst not in self.settings: 
                    self.settings['host_grid_'+inst] = 'default'
                    
                if companion+'_grid_'+inst not in self.settings: 
                    self.settings[companion+'_grid_'+inst] = 'default'
                    
                if is_empty_or_none('host_ld_law_'+inst): 
                    self.settings['host_ld_law_'+inst] = None
                    
                if is_empty_or_none(companion+'_ld_law_'+inst):
                    self.settings[companion+'_ld_law_'+inst] = None
 
                if is_empty_or_none('host_ld_space_'+inst): 
                    self.settings['host_ld_space_'+inst] = 'q'
                    
                if is_empty_or_none(companion+'_ld_space_'+inst):
                    self.settings[companion+'_ld_space_'+inst] = 'q'        
                    
                if 'host_shape_'+inst not in self.settings: 
                    self.settings['host_shape_'+inst] = 'sphere'
                    
                if companion+'_shape_'+inst not in self.settings: 
                    self.settings[companion+'_shape_'+inst] = 'sphere'
                    
                    
        for companion in self.settings['companions_rv']:
            for inst in list(self.settings['inst_rv']) + list(self.settings['inst_rv2']):
                if companion+'_flux_weighted_'+inst in self.settings: 
                    self.settings[companion+'_flux_weighted_'+inst] = set_bool(self.settings[companion+'_flux_weighted_'+inst])
                else:
                    self.settings[companion+'_flux_weighted_'+inst] = False
        
    
        if 'exact_grav' in self.settings: 
            self.settings['exact_grav'] = set_bool(self.settings['exact_grav'])
        else:
            self.settings['exact_grav'] = False
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Phase curve styles
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if is_empty_or_none('phase_curve_style'):
            self.settings['phase_curve_style'] = None
        if self.settings['phase_curve_style'] not in [None, 'sine_series', 'sine_physical', 'ellc_physical', 'GP']:
            raise ValueError("The setting 'phase_curve_style' must be one of [None, 'sine_series', 'sine_physical', 'ellc_physical', 'GP'], but was '"+str(self.settings['phase_curve_style'])+"'.")
        if (self.settings['phase_curve'] is True) and (self.settings['phase_curve_style'] is None):
            raise ValueError("You chose 'phase_curve=True' but did not select a 'phase_curve_style'; please select one of ['sine_series', 'sine_physical', 'ellc_physical', 'GP'].")
        if (self.settings['phase_curve'] is False) and (self.settings['phase_curve_style'] in ['sine_series', 'sine_physical', 'ellc_physical', 'GP']):
           raise ValueError("You chose 'phase_curve=False' but also selected a 'phase_curve_style'; please double check and set 'phase_curve_style=None' (or remove it).")
                               
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Stellar variability
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for key in ['flux', 'rv', 'rv2']:
            if ('stellar_var_'+key not in self.settings) or (self.settings['stellar_var_'+key] is None) or (self.settings['stellar_var_'+key].lower()=='none'): 
                self.settings['stellar_var_'+key] = 'none'
                     
                     
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Baselines
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_all']:
            if inst in self.settings['inst_phot']: key='flux'
            elif inst in self.settings['inst_rv']: key='rv'
            elif inst in self.settings['inst_rv2']: key='rv2'
            
            #::: default
            #::: if the user gives no baseline, the default is 'none'
            if 'baseline_'+key+'_'+inst not in self.settings: 
                self.settings['baseline_'+key+'_'+inst] = 'none'

            #::: hybrid_spline
            #::: the user can define the s value directly, e.g. as "hybrid_spline 0.001"
            #::: this block serves to split up this input and assign it to the right functions
            if ('hybrid_spline' in self.settings['baseline_'+key+'_'+inst])\
                and (len(self.settings['baseline_'+key+'_'+inst].split(' '))>1): 
                s = self.settings['baseline_'+key+'_'+inst].split(' ')[1]
                self.settings['baseline_'+key+'_'+inst] = 'hybrid_spline_s'
                self.settings['baseline_'+key+'_'+inst+'_args'] = s #any arguments coming with this baseline (for future expandability; for now it is simply the s-value)
                
            #::: sample_GP
            #::: make sure the keywords are updated correctly
            elif self.settings['baseline_'+key+'_'+inst] == 'sample_GP': 
                 warnings.warn('You are using outdated keywords. Automatically renaming sample_GP ---> sample_GP_Matern32.'+'. Please update your files before the Duolingo owl comes to get you.') #, category=DeprecationWarning)
                 self.settings['baseline_'+key+'_'+inst] = 'sample_GP_Matern32'
                 
            #::: baseline against custom series
            #::: allows the user to fit a baseline not vs. time but vs. a chosen custom series
            if 'baseline_'+key+'_'+inst+'_against' not in self.settings:
                self.settings['baseline_'+key+'_'+inst+'_against'] = 'time'
            if self.settings['baseline_'+key+'_'+inst+'_against'] not in ['time','custom_series']:
                raise ValueError("The setting 'baseline_'+key+'_'+inst+'_against' must be one of ['time', custom_series'], but was '" + self.settings['baseline_'+key+'_'+inst+'_against'] + "'.")
                     
                
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Errors
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_all']:
            if inst in self.settings['inst_phot']: key='flux'
            elif inst in self.settings['inst_rv']: key='rv'
            elif inst in self.settings['inst_rv2']: key='rv2'
            if 'error_'+key+'_'+inst not in self.settings: 
                self.settings['error_'+key+'_'+inst] = 'sample'
            
        # for inst in self.settings['inst_phot']:
        #     for key in ['flux']:
        #         if 'error_'+key+'_'+inst not in self.settings: 
        #             self.settings['error_'+key+'_'+inst] = 'sample'

        # for inst in self.settings['inst_rv']:
        #     for key in ['rv']:
        #         if 'error_'+key+'_'+inst not in self.settings: 
        #             self.settings['error_'+key+'_'+inst] = 'sample'
                    
                
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
        #::: Plot zoom window
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'zoom_window' not in self.settings:
            self.settings['zoom_window'] = 8./24. #8h window around transit/eclipse midpoint by Default
        else:
            self.settings['zoom_window'] = float(self.settings['zoom_window'])
            
            
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
        For the full list of options see www.allesfitter.com
        '''
    
        #==========================================================================
        #::: load params.csv
        #==========================================================================   
        buf = np.genfromtxt(os.path.join(self.datadir,'params.csv'), delimiter=',',comments='#',dtype=None,encoding='utf-8',names=True)
            
           
        #==========================================================================
        #::: function to assure backwards compability
        #==========================================================================
        def backwards_compability(key_new, key_deprecated):
            if key_deprecated in np.atleast_1d(buf['name']):
                warnings.warn('You are using outdated keywords. Automatically renaming '+key_deprecated+' ---> '+key_new+'. Please fix this before the Duolingo owl comes to get you.') #, category=DeprecationWarning)
                ind = np.where(buf['name'] == key_deprecated)[0]
                np.atleast_1d(buf['name'])[ind] = key_new
                
                
        #==========================================================================
        #::: luser-proof: backwards compability 
        # (has to happend first thing and right inside buf['name'])
        #==========================================================================
        for inst in self.settings['inst_all']:
            backwards_compability(key_new='host_ldc_q1_'+inst, key_deprecated='ldc_q1_'+inst)
            backwards_compability(key_new='host_ldc_q2_'+inst, key_deprecated='ldc_q2_'+inst)
            backwards_compability(key_new='host_ldc_q3_'+inst, key_deprecated='ldc_q3_'+inst)
            backwards_compability(key_new='host_ldc_q4_'+inst, key_deprecated='ldc_q4_'+inst)
            backwards_compability(key_new='ln_err_flux_'+inst, key_deprecated='log_err_flux_'+inst)
            backwards_compability(key_new='ln_jitter_rv_'+inst, key_deprecated='log_jitter_rv_'+inst)
            backwards_compability(key_new='baseline_gp_matern32_lnsigma_flux_'+inst, key_deprecated='baseline_gp1_flux_'+inst)
            backwards_compability(key_new='baseline_gp_matern32_lnrho_flux_'+inst, key_deprecated='baseline_gp2_flux_'+inst)
            backwards_compability(key_new='baseline_gp_matern32_lnsigma_rv_'+inst, key_deprecated='baseline_gp1_rv_'+inst)
            backwards_compability(key_new='baseline_gp_matern32_lnrho_rv_'+inst, key_deprecated='baseline_gp2_rv_'+inst)
                   
                    
        #==========================================================================
        #::: luser-proof: check for allowed keys to catch typos etc.
        #==========================================================================  
        #TODO
                
                
        #==========================================================================
        #::: set up stuff   
        #==========================================================================          
        self.allkeys = np.atleast_1d(buf['name']) #len(all rows in params.csv)
        self.labels = np.atleast_1d(buf['label']) #len(all rows in params.csv)
        self.units = np.atleast_1d(buf['unit']) #len(all rows in params.csv)
        if 'truth' in buf.dtype.names:
            self.truths = np.atleast_1d(buf['truth']) #len(all rows in params.csv)
        else:
            self.truths = np.nan * np.ones(len(self.allkeys))
            
        self.params = collections.OrderedDict() #len(all rows in params.csv)
        self.params['user-given:'] = '' #just for pretty printing
        for i,key in enumerate(self.allkeys):
            #::: if it's not a "coupled parameter", then use the given value
            if np.atleast_1d(buf['value'])[i] not in list(self.allkeys):
                self.params[key] = np.float(np.atleast_1d(buf['value'])[i])
            #::: if it's a "coupled parameter", then write the string of the key it is coupled to
            else:
                self.params[key] = np.atleast_1d(buf['value'])[i]
                
                
        #==========================================================================
        #::: function to automatically set default params if they were not given
        #==========================================================================
        def validate(key, default, default_min, default_max):
            if (key in self.params) and (self.params[key] is not None):
                if (self.params[key] < default_min) or (self.params[key] > default_max):
                    raise ValueError("User input for "+key+" is "+self.params+" but must lie within ["+str(default_min)+","+str(default_max)+"].")
            if (key not in self.params):
                self.params[key] = default
        
        
        #==========================================================================
        #::: luser-proof: make sure the limb darkening values are uniquely 
        #::: from either the u- or q-space
        #==========================================================================  
        def check_ld(obj, inst):
           if self.settings[obj+'_ld_space_'+inst] == 'q': 
                matches = fnmatch.filter(self.allkeys, obj+'_ldc_u*_'+inst)
                if len(matches) > 0:
                    raise ValueError("The following user input is inconsistent:\n"+\
                                     "Setting: '"+key+"' = 'q'\n"+\
                                     "Parameters: {}".format(matches))   
                        
           elif self.settings[obj+'_ld_space_'+inst] == 'u': 
                matches = fnmatch.filter(self.allkeys, obj+'_ldc_q*_'+inst)
                if len(matches) > 0:
                    raise ValueError("The following user input is inconsistent:\n"+\
                                     "Setting: '"+key+"' = 'u'\n"+\
                                     "Parameters: {}".format(matches))  
                        
        for inst in self.settings['inst_all']:
            for obj in ['host'] + self.settings['companions_all']:   
                check_ld(obj, inst)
            
            
        #==========================================================================
        #::: validate that initial guess params have reasonable values
        #==========================================================================
        self.params['automatically set:'] = '' #just for pretty printing
        for companion in self.settings['companions_all']:
            for inst in self.settings['inst_all']:
                
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: ellc defaults
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                
                #::: frequently used parameters
                validate(companion+'_rr', None, 0., np.inf)
                validate(companion+'_rsuma', None, 0., np.inf)
                validate(companion+'_cosi', 0., 0., 1.)
                validate(companion+'_epoch', 0., -np.inf, np.inf)
                validate(companion+'_period', 0., 0., np.inf)
                validate(companion+'_sbratio_'+inst, 0., 0., np.inf)
                validate(companion+'_K', 0., 0., np.inf)
                validate(companion+'_f_s', 0., -1, 1)
                validate(companion+'_f_c', 0., -1, 1)
                validate('dil_'+inst, 0., -np.inf, np.inf)
                
                #::: limb darkenings, u-space
                validate('host_ldc_u1_'+inst, None, 0, 1)
                validate('host_ldc_u2_'+inst, None, 0, 1)
                validate('host_ldc_u3_'+inst, None, 0, 1)
                validate('host_ldc_u4_'+inst, None, 0, 1)
                validate(companion+'_ldc_u1_'+inst, None, 0, 1)
                validate(companion+'_ldc_u2_'+inst, None, 0, 1)
                validate(companion+'_ldc_u3_'+inst, None, 0, 1)
                validate(companion+'_ldc_u4_'+inst, None, 0, 1)

                #::: limb darkenings, q-space
                validate('host_ldc_q1_'+inst, None, 0, 1)
                validate('host_ldc_q2_'+inst, None, 0, 1)
                validate('host_ldc_q3_'+inst, None, 0, 1)
                validate('host_ldc_q4_'+inst, None, 0, 1)
                validate(companion+'_ldc_q1_'+inst, None, 0, 1)
                validate(companion+'_ldc_q2_'+inst, None, 0, 1)
                validate(companion+'_ldc_q3_'+inst, None, 0, 1)
                validate(companion+'_ldc_q4_'+inst, None, 0, 1)
                
                #::: catch exceptions
                if self.params[companion+'_period'] is None:
                    self.settings['do_not_phase_fold'] = True
                
                #::: advanced parameters
                validate(companion+'_a', None, 0., np.inf)
                validate(companion+'_q', 1., 0., np.inf)
                
                validate('didt_'+inst, None, -np.inf, np.inf)
                validate('domdt_'+inst, None, -np.inf, np.inf)
                
                validate('host_gdc_'+inst, None, 0., 1.)
                validate('host_rotfac_'+inst, 1., 0., np.inf)
                validate('host_hf_'+inst, 1.5, -np.inf, np.inf)
                validate('host_bfac_'+inst, None, -np.inf, np.inf)
                validate('host_heat_'+inst, None, -np.inf, np.inf)
                validate('host_lambda', None, -np.inf, np.inf)
                validate('host_vsini', None, -np.inf, np.inf)
                
                validate(companion+'_gdc_'+inst, None, 0., 1.)
                validate(companion+'_rotfac_'+inst, 1., 0., np.inf)
                validate(companion+'_hf_'+inst, 1.5, -np.inf, np.inf)
                validate(companion+'_bfac_'+inst, None, -np.inf, np.inf)
                validate(companion+'_heat_'+inst, None, -np.inf, np.inf)
                validate(companion+'_lambda', None, -np.inf, np.inf)
                validate(companion+'_vsini', None, -np.inf, np.inf)
        
                #::: special parameters (list type)
                if 'host_spots_'+inst not in self.params:
                    self.params['host_spots_'+inst] = None
                if companion+'_spots_'+inst not in self.params:
                    self.params[companion+'_spots_'+inst] = None
                    
                
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: errors and jitters
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #TODO: add validations for all errors / jitters
                    
                
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: baselines (and backwards compability)
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #TODO: add validations for all baseline params
                
                    
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: phase curve style: sine_series
                # all in ppt
                # A1 (beaming)
                # B1 (atmospheric), can be split in thermal and reflected
                # B2 (ellipsoidal)
                # B3 (ellipsoidal 2nd order)
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                # if (self.settings['phase_curve_style'] == 'sine_series') and (inst in self.settings['inst_phot']):
                if (inst in self.settings['inst_phot']):
                    validate(companion+'_phase_curve_A1_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_B1_'+inst, None, -np.inf, 0.)
                    validate(companion+'_phase_curve_B1_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_B1t_'+inst, None, -np.inf, 0.)
                    validate(companion+'_phase_curve_B1t_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_B1r_'+inst, None, -np.inf, 0.)
                    validate(companion+'_phase_curve_B1r_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_B2_'+inst, None, -np.inf, 0.)
                    validate(companion+'_phase_curve_B3_'+inst, None, -np.inf, 0.)

                                       
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: phase curve style: sine_physical  
                # A1 (beaming)
                # B1 (atmospheric), can be split in thermal and reflected
                # B2 (ellipsoidal)
                # B3 (ellipsoidal 2nd order)  
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                # if (self.settings['phase_curve_style'] == 'sine_physical') and (inst in self.settings['inst_phot']):
                if (inst in self.settings['inst_phot']):
                    validate(companion+'_phase_curve_beaming_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_atmospheric_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_atmospheric_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_atmospheric_thermal_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_atmospheric_thermal_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_atmospheric_reflected_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_atmospheric_reflected_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_ellipsoidal_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_ellipsoidal_2nd_'+inst, None, 0., np.inf)
                
                
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: to avoid a bug/feature in ellc, if either property is >0, set the other to 1-15 (not 0):
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                if self.params[companion+'_heat_'+inst] is not None:
                    if (self.params[companion+'_sbratio_'+inst] == 0) and (self.params[companion+'_heat_'+inst] > 0):
                        self.params[companion+'_sbratio_'+inst] = 1e-15        #this is to avoid a bug/feature in ellc
                    if (self.params[companion+'_sbratio_'+inst] > 0) and (self.params[companion+'_heat_'+inst] == 0):
                        self.params[companion+'_heat_'+inst] = 1e-15           #this is to avoid a bug/feature in ellc
              

                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: luser proof: avoid conflicting/degenerate phase curve commands
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                if (inst in self.settings['inst_phot']) and (self.settings['phase_curve'] == True):
                    phase_curve_model_1 = (self.params[companion+'_phase_curve_B1_'+inst] is not None)
                    phase_curve_model_2 = ((self.params[companion+'_phase_curve_B1t_'+inst] is not None) or (self.params[companion+'_phase_curve_B1r_'+inst] is not None))
                    phase_curve_model_3 = (self.params[companion+'_phase_curve_atmospheric_'+inst] is not None)
                    phase_curve_model_4 = ((self.params[companion+'_phase_curve_atmospheric_thermal_'+inst] is not None) or (self.params[companion+'_phase_curve_atmospheric_reflected_'+inst] is not None))
                    phase_curve_model_5 = ((self.params['host_bfac_'+inst] is not None) or (self.params['host_heat_'+inst] is not None) or \
                                           (self.params['host_gdc_'+inst] is not None) or (self.settings['host_shape_'+inst]!='sphere') or \
                                           (self.params[companion+'_bfac_'+inst] is not None) or (self.params[companion+'_heat_'+inst] is not None) or \
                                           (self.params[companion+'_gdc_'+inst] is not None) or (self.settings[companion+'_shape_'+inst]!='sphere'))
                    if (phase_curve_model_1 + phase_curve_model_2 + phase_curve_model_3 + phase_curve_model_4 + phase_curve_model_5) > 1:
                        raise ValueError('You can use either\n'\
                                         +'1) the sine_series phase curve model with "*_phase_curve_B1_*",\n'\
                                         +'2) the sine_series phase curve model with "*_phase_curve_B1t_*" and "*_phase_curve_B1r_*", or\n'\
                                         +'3) the sine_physical phase curve model with "*_phase_curve_atmospheric_*",\n'\
                                         +'4) the sine_physical phase curve model with "*_phase_curve_atmospheric_thermal_*" and "*_phase_curve_atmospheric_reflected_*", or\n'\
                                         +'5) the ellc_physical phase curve model with "*_bfac_*", "*_heat_*", "*_gdc_*" etc.\n'\
                                         +'but you shall not pass with a mix&match.')
                    
                        
        #==========================================================================
        #::: coupled params
        #==========================================================================
        if 'coupled_with' in buf.dtype.names:
            self.coupled_with = buf['coupled_with']
        else:
            self.coupled_with = [None]*len(self.allkeys)
            
        for i, key in enumerate(self.allkeys):
            if isinstance(self.coupled_with[i], str) and (len(self.coupled_with[i])>0):
                self.params[key] = self.params[self.coupled_with[i]]           #luser proof: automatically set the values of the params coupled to another param
                buf['fit'][i] = 0                                              #luser proof: automatically set fit=0 for the params coupled to another param
        
        
        #==========================================================================
        #::: mark to be fitted params
        #==========================================================================
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

    
        #==========================================================================
        #::: luser proof: check if all initial guesses lie within their bounds
        #==========================================================================
        #TODO: make this part of the validate() function
        for th, b, key in zip(self.theta_0, self.bounds, self.fitkeys):
                  
            #:::: test bounds
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
                data['TESS']['time'], data['TESS']['flux'], etc.
            A RV curve is stored as
                data['HARPS']['time'], data['HARPS']['flux'], etc.
        '''
        self.fulldata = {}
        self.data = {}
        
        #======================================================================
        #::: photometry
        #======================================================================
        for inst in self.settings['inst_phot']:
            try:
                time, flux, flux_err, custom_series = np.genfromtxt(os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:4]     
            except:
                time, flux, flux_err = np.genfromtxt(os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:3]     
                custom_series = np.zeros_like(time)
            if any(np.isnan(time*flux*flux_err*custom_series)):
                raise ValueError('There are NaN values in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            if any(flux_err==0):
                raise ValueError('There are uncertainties with values of 0 in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            if any(flux_err<0):
                raise ValueError('There are uncertainties with negative values in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
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
                          'err_scales_flux':flux_err/np.nanmean(flux_err),
                          'custom_series':custom_series
                         }
            if (self.settings['fast_fit']) and (len(self.settings['inst_phot'])>0): 
                time, flux, flux_err, custom_series = self.reduce_phot_data(time, flux, flux_err, custom_series=custom_series, inst=inst)
            self.data[inst] = {
                          'time':time,
                          'flux':flux,
                          'err_scales_flux':flux_err/np.nanmean(flux_err),
                          'custom_series':custom_series
                         }
            
        #======================================================================
        #::: RV
        #======================================================================
        for inst in self.settings['inst_rv']:
            try:
                time, rv, rv_err, custom_series = np.genfromtxt( os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:4]       
            except:
                time, rv, rv_err = np.genfromtxt( os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:3]              
                custom_series = np.zeros_like(time)
            if any(np.isnan(time*rv*rv_err*custom_series)):
                raise ValueError('There are NaN values in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            #aCkTuaLLLyy rv_err=0 is ok, since we add a jitter term here anyway (instead of scaling)
            # if any(rv_err==0):
            #     raise ValueError('There are uncertainties with values of 0 in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            if any(rv_err<0):
                raise ValueError('There are uncertainties with negative values in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            if not all(np.diff(time)>0):
                raise ValueError('Your time array in "'+inst+'.csv" is not sorted. You will want to check that...')
            self.data[inst] = {
                          'time':time,
                          'rv':rv,
                          'white_noise_rv':rv_err,
                          'custom_series':custom_series
                         }
            
        #======================================================================
        #::: RV2 (for detached binaries)
        #======================================================================
        for inst in self.settings['inst_rv2']:
            try:
                time, rv, rv_err, custom_series = np.genfromtxt( os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:4]       
            except:
                time, rv, rv_err = np.genfromtxt( os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:3]              
                custom_series = np.zeros_like(time)
            if not all(np.diff(time)>0):
                raise ValueError('Your time array in "'+inst+'.csv" is not sorted. You will want to check that...')
            self.data[inst] = {
                          'time':time,
                          'rv2':rv,
                          'white_noise_rv2':rv_err,
                          'custom_series':custom_series
                         }
        
        #======================================================================
        #::: also save the combined time series
        #::: for cases where all instruments are treated together
        #::: e.g. for stellar variability GPs
        #======================================================================
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
        self.data['inst_rv']['rv_err'] = np.array(self.data['inst_rv']['rv_err'])[ind_sort]   
        self.data['inst_rv']['inst'] = np.array(self.data['inst_rv']['inst'])[ind_sort]
    
        self.data['inst_rv2'] = {'time':[],'rv2':[],'rv2_err':[],'inst':[]}
        for inst in self.settings['inst_rv2']:
            self.data['inst_rv2']['time'] += list(self.data[inst]['time'])
            self.data['inst_rv2']['rv2'] += list(self.data[inst]['rv2'])
            self.data['inst_rv2']['rv2_err'] += list(np.nan*self.data[inst]['rv2']) #errors will be sampled/derived later
            self.data['inst_rv2']['inst'] += [inst]*len(self.data[inst]['time'])
        ind_sort = np.argsort(self.data['inst_rv2']['time'])
        self.data['inst_rv2']['ind_sort'] = ind_sort
        self.data['inst_rv2']['time'] = np.array(self.data['inst_rv2']['time'])[ind_sort]
        self.data['inst_rv2']['rv2'] = np.array(self.data['inst_rv2']['rv2'])[ind_sort]
        self.data['inst_rv2']['rv2_err'] = np.array(self.data['inst_rv2']['rv2_err'])[ind_sort]   
        self.data['inst_rv2']['inst'] = np.array(self.data['inst_rv2']['inst'])[ind_sort]

        
            
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
        
        self.logprint('\nShifting epochs into the data center')
        self.logprint('------------------------------------')
        
        #::: for all companions
        for companion in self.settings['companions_all']:
            
            self.logprint('Companion',companion)
            self.logprint('\tinput epoch:',self.params[companion+'_epoch'])
            
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
            try:
                ind_e = np.where(self.fitkeys==companion+'_epoch')[0][0]
                ind_p = np.where(self.fitkeys==companion+'_period')[0][0]
                N_truth_shift = int(np.round((self.settings['mid_epoch']-self.fittruths[ind_e])/self.fittruths[ind_p]))
                self.fittruths[ind_e] += N_truth_shift * self.fittruths[ind_p]
            except:
                pass
            
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
                    
        
                self.logprint('\tshifted epoch:',self.params[companion+'_epoch'])
                self.logprint('\tshifted by',N_shift,'periods')
                


    ###############################################################################
    #::: reduce_phot_data
    ###############################################################################
    def reduce_phot_data(self, time, flux, flux_err, custom_series=None, inst=None):
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
        if custom_series is None: 
            return time, flux, flux_err
        else:
            custom_series = custom_series[ind_in]
            return time, flux, flux_err, custom_series
    
    
    
    ###############################################################################
    #::: setup TTV fit (if chosen)
    ###############################################################################
    def setup_ttv_fit(self):
        '''
        this must be run *after* reduce_phot_data()
        '''
        
        #::: the window we choose to look for transits is determined by fast_fit_width
        window = self.settings['fast_fit_width']
        
        #::: for each companion, stitch together all the time stamps observed by all photometric instruments
        #::: and check which of these times overlap with a potential transit window (determined by fast_fit_width)
        for companion in self.settings['companions_phot']:
            times_combined = []
            for inst in self.settings['inst_phot']:
                times_combined += list(self.data[inst]['time'])
            times_combined = np.sort(times_combined)
            
            self.data[companion+'_tmid_observed_transits'] = get_tmid_observed_transits(times_combined,
                                                                                        self.params[companion+'_epoch'],
                                                                                        self.params[companion+'_period'],
                                                                                        window)
            
            for inst in self.settings['inst_phot']:
                time = self.data[inst]['time']
                for i, t in enumerate(self.data[companion+'_tmid_observed_transits']):
                    ind = np.where((time >= (t - window/2.)) & (time <= (t + window/2.)))[0]
                    self.data[inst][companion+'_ind_time_transit_'+str(i+1)] = ind
                    self.data[inst][companion+'_time_transit_'+str(i+1)] = time[ind]
                    

            #::: THE FOLLOWING PART MOVED INTO THE SEPARATE SCRIPT "PREPARE_TTV_FIT.PY"
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
            
            