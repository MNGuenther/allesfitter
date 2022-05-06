#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:15:26 2020

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
from astropy.stats import sigma_clip as sigma_clip_

#::: specific modules
try:
    from wotan import flatten
except ImportError:
    pass

#::: my modules
from .exoworlds_rdx.lightcurves.lightcurve_tools import rebin_err

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




###############################################################################
#::: clean a time series
###############################################################################
def clean(time, y, y_err=None):
    """
    Cleans all input from masks, NaN or Inf values and returns them as np.ndarray.
    Careful: This changes the length of the arrays!

    Parameters
    ----------
    time : array of flaot
        Time stamps.
    y : array of float
        Any data corresponding to the time stamps.
    y_err : array of float or None, optional
        Error of the data. The default is None.

    Returns
    -------
    Cleaned time, y, and y_err (shorter arrays!)
    
    Explanation
    -----------
    The input might be of a mixutre of types, such as list, np.ndarray, or np.ma.core.MaskedArray.
    For stability, first convert them all to np.ma.core.MaskedArray.
    This will preserve all information, even non-masked NaN and Inf values.
    Then mask all invalid values, retrieve the mask, and use it to remove those values.
    Finally, convert them to a np.ndarray.
    Tested on the following example:
        input:
            time = np.linspace(1,6,6)
            flux = np.ma.array([1,2,3,4,np.nan,np.inf], 
                               mask=[True,True,False,False,False,False])
            flux_err = None
        returns:
            (array([3., 4.]), array([3., 4.]), None)
    """

    time = np.ma.array(time)
    y = np.ma.array(y)
    if y_err is None:
        mask = np.ma.masked_invalid(time*y).mask
    else:
        y_err = np.ma.array(y_err)
        mask = np.ma.masked_invalid(time*y*y_err).mask
        y_err = np.array(y_err[~mask])
    time = np.array(time[~mask])
    y = np.array(y[~mask])
        
    return time, y, y_err
        


###############################################################################
#::: sort a time series
###############################################################################
def sort(time, y, y_err=None):
    """
    Sorts all input in time; the input should be cleaned first.

    Parameters
    ----------
    time : array of flaot
        Time stamps.
    y : array of float
        Any data corresponding to the time stamps.
    y_err : array of float or None, optional
        Error of the data. The default is None.

    Returns
    -------
    Sorted time, y, and y_err.
    """
    ind_sort = np.argsort(time)
    time = time[ind_sort]
    y = y[ind_sort]
    if y_err is not None:
        y_err = y_err[ind_sort]
        
    return time, y, y_err



###############################################################################
#::: sigma clip a time series
###############################################################################
def sigma_clip(time, y, low=4, high=4, return_mask=False):
    """
    Astropy's sigma_clip but returning NaN instead of a masked array.

    Parameters
    ----------
    time : array of flaot
        Time stamps.
    y : array of float
        Any data corresponding to the time stamps.
    low : float, optional
        The lower sigma. The default is 5.
    high : float, optional
        The upper sigma. The default is 5.
    return_mask : bool, optional
        Return the masks or only the clipped time series. The default is False.

    Returns
    -------
    Clipped y (outliers replaced with NaN).
    """
    y2 = sigma_clip_(np.ma.masked_invalid(y), sigma_lower=low, sigma_upper=high) #astropy wants masked arrays
    mask = y2.mask
    y2 = np.array(y2.filled(np.nan)) #use NaN instead of masked arrays, because masked arrays drive me crazy
    
    if not return_mask:
        return y2
    
    else: 
        with np.testing.suppress_warnings() as sup:
            sup.filter(UserWarning)
            mask_upper = (y > np.nanmedian(y2)) * mask
            mask_lower = (y < np.nanmedian(y2)) * mask
        return y2, mask, mask_upper, mask_lower



###############################################################################
#::: slide clip a time series
###############################################################################
def slide_clip(time, y, window_length=1, low=4, high=4, return_mask=False):
    """
    Slide clip outliers from a non-stationary time series;
    a much faster alternative to Wotan's built-in slide clip.

    Parameters
    ----------
    time : array of flaot
        Time stamps.
    y : array of float
        Any data corresponding to the time stamps.\
    window_length : float, optional
        The length of the sliding window (in time's units). The default is 1.
    low : float, optional
        The lower sigma. The default is 5.
    high : float, optional
        The upper sigma. The default is 5.
    return_mask : bool, optional
        Return the masks or only the clipped time series. The default is False.

    Returns
    -------
    Clipped y (outliers replaced with NaN).
    """
    y_flat = flatten(time, y, method='biweight', window_length=window_length)
    y2, mask, mask_upper, mask_lower = sigma_clip(time, y_flat, low=low, high=high, return_mask=True)
    y3 = 1*y
    y3[mask] = np.nan
    
    if not return_mask:
        return y3
    
    else:
        return y3, mask, mask_upper, mask_lower
    
    

###############################################################################
#::: binning
###############################################################################
def binning(time, y, y_err=None, dt=None):
    """
    Bin a time series to bin-widths specified by dt.
    This method also handles gaps and error bars properly.
    Careful: This will change the length of the arrays!

    Parameters
    ----------
    time : array of flaot
        Time stamps.
    y : array of float
        Any data corresponding to the time stamps.
    y_err : array of float or None, optional
        Error of the data. The default is None.
    dt : float or None, optional
        Bin-widths in time's units. The default is None (no binning).

    Returns
    -------
    Binned time, y, and y_err (shorter arrays!).
    """
    time, y, y_err = clean(time, y, y_err)
    if dt is not None:
        return rebin_err(time, y, y_err, dt=dt, 
                         ferr_type='medsig', ferr_style='sem', sigmaclip=True)[0:3]
    else:
        return time, y, y_err



###############################################################################
#::: mask bad data regions
###############################################################################
def mask_regions(time, y, bad_regions=None):
    """
    Mask regions by filling y and y_err with NaN for those selected times.
    
    Parameters
    ----------
    time : array of flaot
        Time stamps.
    y : array of float
        Any data corresponding to the time stamps.
    y_err : array of float or None, optional
        Error of the data. The default is None.
    bad_regions : list or None, optional
        List of tuples like [(start0,end0),(start1,end1),...], 
        where any (start,end) are the start and end points of bad data bad_regions. 
        The default is None.

    Returns
    -------
    Masked time, y, and y_err (bad_regions of y and y_err replaced with NaN)
    """
    y2 = 1.*y
    
    if bad_regions is not None:
        for bad_region in bad_regions:
            ind_bad = np.where((time>=bad_region[0]) & (time<=bad_region[1]))[0]
            y2[ind_bad] = np.nan
    
    return y2