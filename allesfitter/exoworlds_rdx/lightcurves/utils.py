# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:07:00 2016

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
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
from pprint import pprint



###########################################################################
#::: Statistics and time series helpers
###########################################################################

        
def medsig(a):
    '''Compute median and MAD-estimated scatter of array a'''
    med = np.nanmedian(a)
    sig = 1.48 * np.nanmedian(abs(a-med))
    return med, sig   
    
  
  
def running_mean(x, N):
    x[np.isnan(x)] = 0. #reset NAN to 0 to calculate the cumulative sum; mimics the 'pandas' behavior
    cumsum = np.cumsum(np.insert(x, 0., 0.)) 
    return 1.*(cumsum[N:] - cumsum[:-N]) / N 
    


# 'running_median' DOES NOT AGREE WITH THE PANDAS IMPLEMENTATION 'running_median_pandas'
#def running_median(x, N):
#    return np.array(list(RunningMedian(N, x)))


    
def running_mean_pandas(x, N):
    ts = pd.Series(x).rolling(window=N, center=False).mean()
    return ts[~np.isnan(ts)].as_matrix()



def running_median_pandas(x, N):
    ts = pd.Series(x).rolling(window=N, center=False).median()
    return ts[~np.isnan(ts)].as_matrix()


    
def mask_ranges(x, x_min, x_max):
    """"
    Crop out values and indices out of an array x for multiple given ranges x_min to x_max.
    
    Input:
    x: array, 
    x_min: lower limits of the ranges
    x_max: upper limits of the ranges
    
    Output:
    
    
    Example:
    x = np.arange(200)    
    x_min = [5, 25, 90]
    x_max = [10, 35, 110]
    """

    mask = np.zeros(len(x), dtype=bool)
    for i in range(len(x_min)): 
        mask = mask | ((x >= x_min[i]) & (x <= x_max[i]))
    ind_mask = np.arange(len(mask))[mask]
    
    return x[mask], ind_mask, mask 




###########################################################################
#::: Text formatting for tables and plots
###########################################################################



def mystr(x,digits=0):
    if np.isnan(x): return '.'
    elif digits==0: return str(int(round(x,digits)))
    else: return str(round(x,digits))



def format_2sigdigits(x1, x2, x3, nmax=3):
    n = int( np.max( [ -np.floor(np.log10(np.abs(x))) for x in [x1, x2, x3] ] ) + 1 )
    scaling = 0
    extra = None
    if n > nmax:
        scaling = n-1
        n = 1
        extra = r"\cdot 10^{" + str(-scaling) + r"}"
    return str(round(x1*10**scaling, n)).ljust(n+2, '0'), str(round(x2*10**scaling, n)).ljust(n+2, '0'), str(round(x3*10**scaling, n)).ljust(n+2, '0'), extra



def deg2hmsdms(ra, dec):
    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
    return c.to_string('hmsdms', precision=2, sep=':')



def format_latex(x1, x2, x3, nmax=3):
    r, l, u, extra = format_2sigdigits(x1, x2, x3, nmax)
    if l==u:
        core = r + r"\pm" + l
    else:
        core = r + r"^{+" + l + r"}_{-" + u + r"}"

    if extra is None:
        return r"$" + core + r"$"
    else:
        return r"$" + '(' + core + ')' + extra + r"$"



###########################################################################
#::: Dictionaries and tables
###########################################################################


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z
        
        
def table_view(dic):
    from astropy.table import Table 
    dic_table = {}
    subkeys = ['OBJ_ID', 'SYSREM_FLUX3_median', 'PERIOD', 'DEPTH', 'WIDTH', 'NUM_TRANSITS']
    for key in subkeys:
            dic_table[key] = dic[key]
    dic_table = Table(dic_table)
    dic_table = dic_table[subkeys]
    pprint(dic_table)
