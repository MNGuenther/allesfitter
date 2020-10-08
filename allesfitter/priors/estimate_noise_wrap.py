import os, csv
import numpy as np
from .. import config
from . import estimate_noise
from allesfitter.utils.latex_printer import round_txt_separately

def estimate_noise_wrap(pathdata):
    
    print('\nEstimating errors and baselines... this will take a couple of minutes. Please be patient, you will get notified once everything is completed.\n')
    
    #::: run MCMC fit to estimate errors and baselines
    estimate_noise.estimate_noise(pathdata)
    
    def fwrite_params(key, label, unit, physical_bounds):
        if INPUT[key+'_bounds_type'].value == 'uniform':
            bounds = 'uniform ' \
                     + str( np.max( [physical_bounds[0], float(INPUT[key+'_median'].value)-float(INPUT[key+'_lerr'].value)] ) ) + ' ' \
                     + str( np.min( [physical_bounds[1], float(INPUT[key+'_median'].value)+float(INPUT[key+'_uerr'].value)] ) )
        elif INPUT[key+'_bounds_type'].value == 'uniform * 5':
            bounds = 'uniform ' \
                     + str( np.max( [physical_bounds[0], float(INPUT[key+'_median'].value)-5*float(INPUT[key+'_lerr'].value)] ) ) + ' ' \
                     + str( np.min( [physical_bounds[1], float(INPUT[key+'_median'].value)+5*float(INPUT[key+'_uerr'].value)] ) )
        elif INPUT[key+'_bounds_type'].value == 'trunc_normal':
            bounds = 'trunc_normal ' \
                     + str(physical_bounds[0]) + ' ' \
                     + str(physical_bounds[1]) + ' ' \
                     + str(INPUT[key+'_median'].value) + ' ' \
                     + str(np.max( [ float(INPUT[key+'_lerr'].value), float(INPUT[key+'_uerr'].value) ] ))
        elif INPUT[key+'_bounds_type'].value == 'trunc_normal * 5':
            bounds = 'trunc_normal ' \
                     + str(physical_bounds[0]) + ' ' \
                     + str(physical_bounds[1]) + ' ' \
                     + str(INPUT[key+'_median'].value) + ' ' \
                     + str(5*np.max( [ float(INPUT[key+'_lerr'].value), float(INPUT[key+'_uerr'].value) ] ))
        fwrite(key + ',' + str(INPUT[key+'_median'].value) + ',' + str(int(INPUT[key+'_fit'].value)) + ',' +  bounds + ',' + label + ',' + unit)
    
    
    def fwrite(text):
        fname_params = os.path.join(pathdata, 'params.csv')
        with open(fname_params, 'a') as f:
            f.write(text+'\\n')
    
    
    def get_median_and_error_strings(text_median, text_lerr, text_uerr):
        if (text_median.value == ''):
            median = 'NaN'
            nan_fields = True
        else:
            median = text_median.value
        if (text_lerr.value == '') or (text_uerr.value == ''):
            err = 'NaN'
            nan_fields = True
        else:
            err = str( 5.* np.max( [float(text_lerr.value), float(text_uerr.value)] ) )
        median, err, _ = round_txt_separately( float(median), float(err), float(err) )
        return median, err
         
    
    def clean_up_csv(fname, N_last_rows=0):
        
        with open(fname, "r") as f:
            params_csv = list(csv.reader(f))
    
        with open(fname, "w") as f:
            writer = csv.writer(f)
            for i in range(len(params_csv)-N_last_rows):
                row = params_csv[i]
                writer.writerow(row)
    
    
    
    #::: delete the rows containing the default (zero) errors and baselines from the params.csv file
    N_default_rows = 0
    clean_up_csv( os.path.join( pathdata, 'params.csv' ), N_last_rows=N_default_rows )
    
    config.init(pathdata)
        
    #::: write new rows into params.csv
    #::: errors
    fwrite('#errors per instrument,')
    
    for i, inst in enumerate(config.BASEMENT.settings['inst_phot']):         
        #::: read in the summary file
        summaryfile = os.path.join( pathdata, 'priors', 'summary_phot.csv' )
        priors2 = np.genfromtxt(summaryfile, names=True, delimiter=',', dtype=None)
        priors = {}
        for key in priors2.dtype.names:
            priors[key] = np.atleast_1d(priors2[key])
    
        median = priors['ln_yerr_median'][i]
        err = 5.*np.max([ float(priors['ln_yerr_ll'][i]), float(priors['ln_yerr_ul'][i]) ])
        median, err, _ = round_txt_separately(median,err,err)
        fwrite('ln_err_flux_'+inst+','+median+',1,trunc_normal -23 0 '+median+' '+err+',$\log{\sigma_\mathrm{'+inst+'}}$,')
    
    for i, inst in enumerate(config.BASEMENT.settings['inst_rv']):   
        #::: read in the summary file
        summaryfile = os.path.join( pathdata, 'priors', 'summary_rv.csv' )
        priors2 = np.genfromtxt(summaryfile, names=True, delimiter=',', dtype=None)
        priors = {}
        for key in priors2.dtype.names:
            priors[key] = np.atleast_1d(priors2[key])
    
        median = priors['ln_yerr_median'][i]
        err = 5.*np.max([ float(priors['ln_yerr_ll'][i]), float(priors['ln_yerr_ul'][i]) ])
        median, err, _ = round_txt_separately(median,err,err)
        fwrite('ln_jitter_rv_'+inst+','+median+',1,trunc_normal -23 0 '+median+' '+err+',$\log{\sigma_\mathrm{jitter; '+inst+'}}$,')
    
    
    #::: write new rows into params.csv
    #::: baselines
    fwrite('#baseline per instrument,')
    
    for i, inst in enumerate(config.BASEMENT.settings['inst_phot']):         
        #::: read in the summary file
        summaryfile = os.path.join( pathdata, 'priors', 'summary_phot.csv' )
        priors2 = np.genfromtxt(summaryfile, names=True, delimiter=',', dtype=None)
        priors = {}
        for key in priors2.dtype.names:
            priors[key] = np.atleast_1d(priors2[key])
    
        median = priors['gp_ln_sigma_median'][i]
        err = 5.*np.max([ float(priors['gp_ln_sigma_ll'][i]), float(priors['gp_ln_sigma_ul'][i]) ])
        median, err, _ = round_txt_separately(median,err,err)
        fwrite('baseline_gp1_flux_'+inst+','+median+',1,trunc_normal -23 23 '+median+' '+err+',$\mathrm{gp: \log{\sigma} ('+inst+')}$,')
    
        median = priors['gp_ln_rho_median'][i]
        err = 5.*np.max([ float(priors['gp_ln_rho_ll'][i]), float(priors['gp_ln_rho_ul'][i]) ])
        median, err, _ = round_txt_separately(median,err,err)
        fwrite('baseline_gp2_flux_'+inst+','+median+',1,trunc_normal -23 23 '+median+' '+err+',$\mathrm{gp: \log{\\rho} ('+inst+')}$,')





