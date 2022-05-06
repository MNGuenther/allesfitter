#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:00:41 2020

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
import pandas as pd
from scipy.stats import anderson
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson 
from statsmodels.stats.diagnostic import acorr_ljungbox

#::: allesfitter modules
from . import config
from .general_output import logprint

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




def alles_anderson(residuals):
    '''
    Parameters
    ----------
    residuals : array of float
        The residuals after the fit of model+baseline+stellar_var.

    Returns
    -------
    isNormal : bool
        True if the residuals are normally distributed, False otherwise.
    
    Outputs
    -------
    It also prints the statstics and conclusions.
    
    Sauces
    ------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
    https://www.statology.org/anderson-darling-test-python/
    '''
    logprint('Anderson-Darling Test')
    logprint('---------------------')
    logprint('This tests the null hypothesis that the residuals follows a normal distribution.') 
    statistic, critical_values, significance_levels = anderson(residuals)
    logprint('Test statistic\t\t', statistic)
    logprint('Critical values\t\t', critical_values)
    logprint('Significance levels\t', significance_levels/100.)
    logprint('Does the null hypotheses hold at a significance level of...')
    for cv, sl in zip(critical_values, significance_levels):
        logprint('...', sl/100., '\t\t', statistic < cv)
    isNormal = all(statistic < critical_values)
    if isNormal:
        logprint('The null hypothesis cannot be rejected.')
        logprint('In simple words: your residuals look good.')
    else:
        logprint('The null hypothesis is rejected at some significance levels.')
        logprint('In simple words: there might still be some structure in your residuals.')
    logprint('\n')
    
    return isNormal
        


def alles_adfuller(residuals):
    '''
    Parameters
    ----------
    residuals : array of float
        The residuals after the fit of model+baseline+stellar_var.

    Returns
    -------
    isStationary : bool
        True if the residuals are stationary, False otherwise.
    
    Outputs
    -------
    It also prints the statstics and conclusions.
    
    Sauces
    ------
    https://www.hackdeploy.com/augmented-dickey-fuller-test-in-python/
    '''
    class StationarityTests:
        def __init__(self, significance=.05):
            self.SignificanceLevel = significance
            self.pValue = None
            self.isStationary = None
            
        def ADF_Stationarity_Test(self, timeseries, printResults = True):
            logprint('Augmented Dickey-Fuller Test')
            logprint('----------------------------')
            logprint('This tests the null hypothesis that the residuals show non-stationarity (trends).') 
            adfTest = adfuller(timeseries, autolag='AIC')
            self.pValue = adfTest[1]
            if (self.pValue<self.SignificanceLevel):
                self.isStationary = True
            else:
                self.isStationary = False
            if printResults:
                dfResults = pd.Series(adfTest[0:4], index=['Test Statistic','P-Value','# Lags Used','# Observations Used'])
                for key,value in adfTest[4].items():
                    dfResults['Critical Value (%s)'%key] = value
                logprint(dfResults.to_string())
    
    sTest = StationarityTests()
    sTest.ADF_Stationarity_Test(residuals, printResults = True)
    logprint("Is the time series stationary? {0}".format(sTest.isStationary))
    if not sTest.isStationary:
        logprint('In simple words: there might still be some structure in your residuals.')
    else:
        logprint('In simple words: your residuals look good.')
    logprint('\n')
    
    return sTest.isStationary
        
    

def alles_durbin(residuals):
    '''
    Parameters
    ----------
    residuals : array of float
        The residuals after the fit of model+baseline+stellar_var.

    Returns
    -------
    isUncorrelated : bool
        True if the residuals are not correlated, False otherwise.
    
    Outputs
    -------
    It also prints the statstics and conclusions.
    
    Sauces
    ------
    https://www.statology.org/durbin-watson-test-python/
    '''
    statistic = durbin_watson(residuals) 
    isUncorrelated = (statistic>=1.5) & (statistic<=2.5)
    logprint('Durbin-Watson Test')
    logprint('------------------')
    logprint('This tests the null hypothesis that there is no correlation among the residuals.')
    logprint('Test statistic:', statistic)
    logprint('The test statistic is approximately equal to 2*(1-r) where r is the sample autocorrelation of the residuals. Interpretation:')
    logprint('\t< 1.5 suggests a positive correlation')
    logprint('\t1.5 to 2.5 suggests no correlation')
    logprint('\t> 2.5 suggests a negative correlation')
    if isUncorrelated:
        logprint('No signs of a correlation.')
        logprint('In simple words: your residuals look good.')
    elif statistic<1.5:
        logprint('Signs of a positive correlation.')
        logprint('In simple words: there might still be some structure in your residuals.')
    elif statistic>2.5:
        logprint('Signs of a negative correlation.')
        logprint('In simple words: there might still be some structure in your residuals.')
    logprint('\n')

    return isUncorrelated 

    

def alles_ljung(residuals):
    '''
    Parameters
    ----------
    residuals : array of float
        The residuals after the fit of model+baseline+stellar_var.

    Returns
    -------
    isUncorrelated : bool
        True if the residuals are not correlated, False otherwise.
    
    Outputs
    -------
    It also prints the statstics and conclusions.
    
    Sauces
    ------
    https://www.statology.org/ljung-box-test-python/
    https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html?highlight=ljung
    '''
    
    logprint('Ljung-Box Test')
    logprint('--------------')
    logprint('This tests the null hypothesis that there is no correlation among the residuals.')
    
    df = acorr_ljungbox(residuals, lags=[1,5,10,15,20], return_df=True)
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'lag'})
    logprint('Does the null hypotheses hold at a significance level of...')
    df['0.15'] = df['lb_pvalue'] > 0.15 #if Ture, the null hypothesis cannot be rejected at this significance level
    df['0.1'] = df['lb_pvalue'] > 0.10 #if Ture, the null hypothesis cannot be rejected at this significance level
    df['0.05'] = df['lb_pvalue'] > 0.05 #if Ture, the null hypothesis cannot be rejected at this significance level
    df['0.025'] = df['lb_pvalue'] > 0.025 #if Ture, the null hypothesis cannot be rejected at this significance level
    df['0.01'] = df['lb_pvalue'] > 0.01 #if Ture, the null hypothesis cannot be rejected at this significance level
    isUncorrelated = all(df['0.15']==True) & all(df['0.1']==True) & all(df['0.05']==True) & all(df['0.025']==True) & all(df['0.01']==True)
    logprint(df.to_string(index=False))
    if isUncorrelated:
        logprint('The null hypothesis cannot be rejected.')
        logprint('In simple words: your residuals look good.')
    else:
        logprint('The null hypothesis is rejected at some significance levels.')
        logprint('In simple words: there might still be some structure in your residuals.')
    logprint('\n')

    return isUncorrelated 
    
    
    
def residual_stats(residuals):
    '''
    Parameters
    ----------
    residuals : array of float
        The residuals after the fit of model+baseline+stellar_var.
    typ : str
        'mcmc' or 'ns', just givs the name to the output file.

    Returns
    -------
    None.
    
    Outputs
    -------
    Prints the statstics and conclusions.
    '''
    residuals = residuals[ ~np.isnan(residuals) & np.isfinite(residuals) ] #remove all evil
    logprint("\nPerforming diagnostic tests on the fit's residuals...\n")
    passed_anderson = alles_anderson(residuals)
    passed_adfuller = alles_adfuller(residuals)
    passed_durbin = alles_durbin(residuals)
    try:
        passed_ljung = alles_ljung(residuals)
    except:
        logprint('Ljung-Box Test crashed.')
        passed_ljung = '(crashed)'
    logprint('Summary')
    logprint('-------')
    logprint('Test                    Passed?')
    logprint('Anderson-Darling       ',passed_anderson)
    logprint('Augmented Dickey-Fuller',passed_adfuller)
    logprint('Durbin-Watson          ',passed_durbin)
    logprint('Ljung-Box              ',passed_ljung)
    
    
    
# if __name__ == "__main__":
#     residuals = np.random.normal(size=1000)
#     alles_residual_stats(residuals)