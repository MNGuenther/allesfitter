# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:18:27 2016

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
import matplotlib.pyplot as plt

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




######################################################################
# BINNING WITHOUT TIME GAPS 
# !!! DO NOT USE FOR COMBINING DIFFERENT NIGHTS !!!
######################################################################

def binning1D(arr, bin_width, setting='mean', normalize=False):
    """ WARNING: this does not respect boundaries between different night; 
    will average data from different nights"""
    N_time = len(arr)
    N_bins = np.int64(np.ceil(1.*N_time / bin_width))
    binarr, binarr_err = np.zeros((2,N_bins))
    bin_width = int(bin_width)
    if setting=='mean':
        for nn in range(N_bins):
            binarr[nn] = np.nanmean(arr[nn*bin_width:(nn+1)*bin_width])
            binarr_err[nn] = np.nanstd(arr[nn*bin_width:(nn+1)*bin_width])
    if setting=='median':
        for nn in range(N_bins):
            binarr[nn] = np.nanmedian(arr[nn*bin_width:(nn+1)*bin_width])
            binarr_err[nn] = 1.48 * np.nanmedian(abs(arr[nn*bin_width:(nn+1)*bin_width] - binarr[nn]))
    if normalize==True:
        med = np.nanmedian(binarr)
        binarr /= med
        binarr_err /= med
        
    return binarr, binarr_err
    
    
    
def binning2D(arr, bin_width, setting='mean', normalize=False, axis=1):
    #arr being 2D array, with objs on x and time stamps on y
    """ WARNING: this does not respect boundaries between different night; 
    will average data from different nights"""
    N_time = arr.shape[1]
#    print N
    N_objs = arr.shape[0]
#    print N_objs
    N_bins = np.int64(np.ceil(1.*N_time / bin_width))
#    print N_bins
    binarr, binarr_err = np.zeros((2,N_objs,N_bins))
#    print arr.shape
#    print binarr.shape
    bin_width = int(bin_width)
    if setting=='mean':
        for nn in range(N_bins):
            binarr[:,nn] = np.nanmean(arr[:,nn*bin_width:(nn+1)*bin_width], axis=axis)
            binarr_err[:,nn] = np.nanstd(arr[:,nn*bin_width:(nn+1)*bin_width], axis=axis)
    if setting=='median':
        for nn in range(N_bins):
            binarr[:,nn] = np.nanmedian(arr[:,nn*bin_width:(nn+1)*bin_width], axis=axis)
            binarr_err[:,nn] = 1.48 * np.nanmedian(abs(arr[:,nn*bin_width:(nn+1)*bin_width] - binarr[:,nn]))
    if normalize==True:
        med = np.nanmedian(binarr)
        binarr /= med
        binarr_err /= med
        
#    print arr.shape
#    print binarr.shape
        
    return binarr, binarr_err
    
    

######################################################################
# BINNING WITH TIME GAPS 
# !!! USE THIS FOR COMBINING DIFFERENT NIGHTS !!!
######################################################################   
  
def bin_edge_indices(time1D, bin_width, timegap, N_time):
    """ DETERMINE ALL THE BIN-EDGE-INDICES (TO NOT BIN OVER DIFFERENT NIGHTS)"""
    """ this currently relies on the fact that timestamps for all are approximately the same 
    (given for the case of a HJD array that represents MJD values with small corrections)"""

#    ind_start_of_night = np.append( 0 , np.where( np.diff(time) > timegap )[0] + 1 )
    ind_end_of_night = np.append( np.where( np.diff(time1D) > timegap )[0], len(np.diff(time1D)-1 ) )
    N_nights = len(ind_end_of_night)
    
    first_ind = [0]
    last_ind = []
    i = 0    
#    j = 0
    while ((first_ind[-1] < N_time) & (i < N_nights) ):
        if (first_ind[-1]+bin_width) < ind_end_of_night[i]:
            last_ind.append( first_ind[-1] + bin_width )
        else:
            last_ind.append( ind_end_of_night[i] )
            i += 1
        first_ind.append( last_ind[-1] + 1 )
#        j += 1
    
    del first_ind[-1]  
    
    return first_ind, last_ind
    


def binning1D_per_night(time, arr, bin_width, timegap=3600, setting='mean', normalize=False):
    """ If time and arr are 1D arrays """
    
    N_time = len(arr)
    bin_width = int(bin_width)
    
    first_ind, last_ind = bin_edge_indices(time, bin_width, timegap, N_time)
    
    N_bins = len(first_ind)
    bintime, binarr, binarr_err = np.zeros((3,N_bins)) * np.nan
    
    if setting=='mean':
        for nn in range(N_bins):
            #skip no/single data points
            if last_ind[nn] > first_ind[nn]:
                bintime[nn] = np.nanmean( time[first_ind[nn]:last_ind[nn]] )
                #skip All-NAN slices (i.e. where all flux data is masked)
                if ( np.isnan(arr[first_ind[nn]:last_ind[nn]]).all() == False ):
                    binarr[nn] = np.nanmean( arr[first_ind[nn]:last_ind[nn]] )
                    binarr_err[nn] = np.nanstd( arr[first_ind[nn]:last_ind[nn]] )
    elif setting=='median':
        for nn in range(N_bins):
            #skip no/single data points
            if (last_ind[nn] > first_ind[nn]): 
                bintime[nn] = np.nanmedian( time[first_ind[nn]:last_ind[nn]] )
                #skip All-NAN slices (i.e. where all flux data is masked)
                if ( np.isnan(arr[first_ind[nn]:last_ind[nn]]).all() == False ):
                    binarr[nn] = np.nanmedian( arr[first_ind[nn]:last_ind[nn]] )
                    binarr_err[nn] = 1.48 * np.nanmedian( abs(arr[first_ind[nn]:last_ind[nn]] - binarr[nn]) )
    if normalize==True:
        med = np.nanmedian(binarr)
        binarr /= med
        binarr_err /= med
            
    return bintime, binarr, binarr_err     
    


def binning2D_per_night(time, arr, bin_width, timegap=3600, setting='mean', normalize=False, axis=1):
    """ If time and arr are each a 2D array, with different objs on x and different time stamps on y"""
    """ this currently relies on the fact that timestamps for all are approximately the same 
    (given for the case of a HJD array that represents MJD values with small corrections)"""
    N_time = arr.shape[1]
    N_objs = arr.shape[0]
    
    bin_width = int(bin_width)
    
    first_ind, last_ind = bin_edge_indices(time[0,:], bin_width, timegap, N_time)
    
    N_bins = len(first_ind)
    bintime, binarr, binarr_err = np.zeros((3,N_objs,N_bins))
    
    if setting=='mean':
        for nn in range(N_bins):
            bintime[:,nn] = np.nanmean( time[:,first_ind[nn]:last_ind[nn]], axis=axis )
            binarr[:,nn] = np.nanmean( arr[:,first_ind[nn]:last_ind[nn]], axis=axis )
            binarr_err[:,nn] = np.nanstd( arr[:,first_ind[nn]:last_ind[nn]], axis=axis )
    elif setting=='median':
        for nn in range(N_bins):
            bintime[:,nn] = np.nanmedian( time[:,first_ind[nn]:last_ind[nn]], axis=axis )
            binarr[:,nn] = np.nanmedian( arr[:,first_ind[nn]:last_ind[nn]], axis=axis )
            binarr_err[:,nn] = 1.48 * np.nanmedian( abs(arr[:,first_ind[nn]:last_ind[nn]] - binarr[:,nn]) )
    
    if normalize==True:
        med = np.nanmedian(binarr)
        binarr /= med
        binarr_err /= med
            
    return bintime, binarr, binarr_err
               

    
def binning1D_per_night_list(time, arr, bin_width, timegap=3600, setting='mean', normalize=False):
    """ different style of program, same application """
    N = len(time)
    bin_width = int(bin_width)
    
    bintime = []
    binarr = []
    binarr_err = []
    
#    ind_start_of_night = np.append( 0 , np.where( np.diff(time) > timegap )[0] + 1 )
    ind_end_of_night = np.append( np.where( np.diff(time) > timegap )[0], len(np.diff(time)-1 ) )
    N_nights = len(ind_end_of_night)
    first_ind = 0
    i = 0
    
    if setting=='mean':
        while ((first_ind < N) & (i < N_nights) ):
            if (first_ind+bin_width) < ind_end_of_night[i]:
                last_ind = first_ind+bin_width
            else:
                last_ind = ind_end_of_night[i]
                i += 1
                
            bintime.append( np.nanmean( time[first_ind:last_ind] ) )
            binarr.append( np.nanmean( arr[first_ind:last_ind] ) )
            binarr_err.append( np.nanstd(arr[first_ind:last_ind]) )
            first_ind = last_ind + 1
            
    elif setting=='median':       
        while first_ind < N:
            if (first_ind+bin_width) < ind_end_of_night[i]:
                last_ind = first_ind+bin_width
            else:
                last_ind = ind_end_of_night[i]
                i += 1
                
            bintime.append( np.nanmedian( time[first_ind:last_ind] ) )
            binarr.append( np.nanmedian( arr[first_ind:last_ind] ) )
            binarr_err.append( 1.48 * np.nanmedian(abs( arr[first_ind:last_ind] - binarr[-1])) )
            first_ind = last_ind

    bintime = np.array(bintime)
    binarr = np.array(binarr)
    binarr_err = np.array(binarr_err)
    
    if normalize==True:
        med = np.nanmedian(binarr)
        binarr /= med
        binarr_err /= med
        
    return bintime, binarr, binarr_err       
    
    

######################################################################
# MAIN (FOR TESTING)
######################################################################
if __name__ == '__main__':
    
    ######################################################################
    # TEST binning2D_per_night
    ######################################################################
    arr = np.array([[1,2,3,4,5,6,  67,68,64,  -10,-11,-13], \
                   [1,2,3,4,5,6,  24,28,32,  10,11,13]])           
    time = np.array([[1,2,3,4,5,6,   10001,10002,10003,    20001,20002,20003], \
                    [1,2,3,4,5,6.1,   10001,10002.1,10003.3,    20001,20002,20003]])
    
    bintime,binarr, _ = binning2D_per_night(time,arr,6)
    
    plt.figure()
    plt.plot(time,arr,'k.')
    plt.plot(bintime,binarr,'r.')
    
    
    ######################################################################
    # TEST binning1D_per_night
    ######################################################################
    arr = np.array([1,2,3,4,5,6,  67,68,64,  -10,-11,-13])         
    time = np.array([1,2,3,4,5,6,   10001,10002,10003,    20001,20002,20003])
                    
    bintime,binarr, _ = binning1D_per_night(time,arr,6)
    
    plt.figure()
    plt.plot(time,arr,'k.')
    plt.plot(bintime,binarr,'r.')
