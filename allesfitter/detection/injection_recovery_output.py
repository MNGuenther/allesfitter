#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:01:30 2019

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
import os
import numpy as np
import matplotlib.pyplot as plt

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

    
    
    
###############################################################################
#::: fct to check if the right signal was found
###############################################################################
def is_multiple_of(a, b, tolerance=0.05):
    a = np.float(a)
    b = np.float(b) 
    result = a % b
    return (abs(result/b) <= tolerance) or (abs((b-result)/b) <= tolerance)


def is_detected(inj_period, inj_epoch, tls_period, tls_epoch, tls_SNR, SNR_threshold):
    a = is_multiple_of(tls_period, inj_period/2.) #check if it is a multiple of half the period to within 5%
    b = ((abs(tls_epoch-inj_epoch)%inj_period) < 0.05) or ((abs(abs(tls_epoch-inj_epoch)%inj_period)-0.5) < 0.05) #check if the epoch's match to within 5% of the period (also allow half-period offsets)
    c = tls_SNR>=SNR_threshold
#    right_epoch = False
#    for tt in results.transit_times:
#        for i in range(-5,5):
#            right_epoch = right_epoch or (np.abs(tt-epoch+i*period) < (1./24.)) #check if any epochs matches to within 1 hour
             
#    right_depth   = (np.abs(np.sqrt(1.-results.depth)*rstar - rplanet)/rplanet < 0.05) #check if the depth matches
                
    if a*b*c:
        return True
    else:
        return False
                    
    
    
def is_detected_list(results, SNR_threshold):
    detected = [ is_detected(results['inj_period'][i], results['inj_epoch'][i], results['tls_period'][i], results['tls_epoch'][i], results['tls_SNR'][i], SNR_threshold) for i in range(len(results))]
    return np.array(detected)
    
    

###############################################################################
#::: plot
###############################################################################
def irplot(fname, SNR_threshold, period_bins=None, rplanet_bins=None, options=None):
    
    #::: handle inputs
    if options is None: options = {}
    if 'cmap' not in options: options['cmap'] = 'Blues_r'
    if 'logfname' not in options: options['logfname'] = 'injection_recovery_test.csv'
    if 'outdir' not in options: options['outdir'] = ''
    if len(options['outdir'])>0 and not os.path.exists(options['outdir']): os.makedirs(options['outdir'])
    
    #::: load the files and check which TLS detection matches an injection; 
    #::: note that one injection will have multiple TLS detections (due not false positives)
    results = np.genfromtxt(fname, delimiter=',', dtype=None, names=True)
    inj_periods = np.atleast_1d(results['inj_period'])
    inj_rplanets = np.atleast_1d(results['inj_rplanet'])
    try: 
        inj_depths = np.atleast_1d(results['inj_depth'])
    except:
        inj_depths = np.nan*inj_rplanets
    # tls_periods = np.atleast_1d(results['tls_period'])
    # tls_SNRs = np.atleast_1d(results['tls_SNR'])
    detected = is_detected_list(results, SNR_threshold)
    # print(detected)
    
    #::: now boil it down to unique injections and see whether any TLS detection matched it
    # period = np.unique(inj_periods)
    # rplanet = np.unique(inj_rplanets)
    period = []
    rplanet = []
    depth = []
    found = []
    for p in np.unique(inj_periods):
        for r in np.unique(inj_rplanets):
            period.append(p)
            rplanet.append(r)
            ind = np.where( (inj_periods==p) & (inj_rplanets==r) )[0]
            f = any( detected[ind] )
            found.append(f)
            depth.append(inj_depths[ind][0])
            # print(p,r,ind,f)
    period = np.array(period)
    rplanet = np.array(rplanet)
    depth = np.array(depth)
    found = np.array(found)
    
    
    
    ###############################################################################
    #::: match
    ###############################################################################
    def get_inj_depth_from_inj_rplanet(r):
        plt.figure()
        plt.plot(np.sort(rplanet), np.sort(depth))
        return np.interp( r, np.sort(rplanet), np.sort(depth) )
    
    
    def get_inj_rplanet_from_inj_depth(d):
        d = np.array(d)
        d[ d>np.nanmax(depth) ] = np.nan
        d[ d<np.nanmin(depth) ] = np.nan
        plt.figure()
        plt.plot(np.sort(rplanet), np.sort(depth))
        return np.interp( d, np.sort(depth), np.sort(rplanet) )
        
    
    
    ###############################################################################
    #::: scatter plot
    ###############################################################################
    if ( len(np.unique(inj_periods)) * len(np.unique(inj_periods)) ) < 100:
        print('\nCreating scatter plot (too few sampled for 2D histogram plot).\n')
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(period, rplanet, c=found, s=100/np.log10(len(period)), cmap=options['cmap'], edgecolors='b')
        ax.set(xlabel='Period (days)', ylabel='Radius '+r'$(R_\oplus)$')
        ax.text(0.5,1.05,'SNR>'+str(SNR_threshold)+' | blue: not recovered | white: recovered',ha='center',va='center',transform=ax.transAxes)
        try: fig.savefig( os.path.join(options['outdir'],options['logfname'].split('.')[0]+'_snr'+str(SNR_threshold)+'_scatter.pdf'), bbox_inches='tight') #some matplotlib versions crash when saving pdf...
        except: fig.savefig( os.path.join(options['outdir'],options['logfname'].split('.')[0]+'_snr'+str(SNR_threshold)+'_scatter.jpg'), bbox_inches='tight') #some matplotlib versions need pillow for jpg (conda install pillow)...
        plt.close(fig)
        # err
    
    
    
    ###############################################################################
    #::: histogram (normed)
    ###############################################################################
    if ( len(np.unique(inj_periods)) * len(np.unique(inj_periods)) ) >= 100:
        print('\nCreating 2D histogram plot (too many samples for scatter plot).\n')
        if (period_bins is not None) and (rplanet_bins is not None):
            bins = [period_bins, rplanet_bins]
        else:
            bins = [np.histogram_bin_edges(period, bins='auto'), np.histogram_bin_edges(rplanet, bins='auto')]
        h1,x,y = np.histogram2d(period[found==1], rplanet[found==1], bins=bins)
        h2,x,y = np.histogram2d(period[found==0], rplanet[found==0], bins=bins)
        normed_hist = (100.*h1/(h1+h2))
        
        fig, ax = plt.subplots(figsize=(6.8,5))
        X, Y = np.meshgrid(x, y)
        im = ax.pcolormesh(X, Y, normed_hist.T, cmap=options['cmap'], vmin=0, vmax=100);
        # im = plt.imshow(normed_hist.T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), interpolation='none', aspect='auto', cmap='jet', vmin=0, vmax=100, rasterized=True)
        plt.colorbar(im, label='Recovery rate (%)', pad=0.12)
        ax.set(xlabel='Injected period (days)', ylabel=r'Injected radius (R$_\oplus$)', title='SNR>'+str(SNR_threshold))
        # change_font(ax)
        
        ax1 = ax.twinx()
        y0, y1 = ax.get_ylim()
        ax1.set(ylabel = 'Observed depth (ppt)',
                # ylim = get_inj_depth_from_inj_rplanet(ax.get_ylim()))
                ylim = ax.get_ylim(), #in Rearth
                yticks = get_inj_rplanet_from_inj_depth([0.1,0.5,1,2,5,10,15,20,50,100]), #tick positions in rearth
                yticklabels = [0.1,0.5,1,2,5,10,15,20,50,100]) #labels in ppt
                # yticks = [0,0.25,0.5,0.75,1],
                # yticklabels = [ np.format_float_positional( get_inj_depth_from_inj_rplanet(y0+(y1-y0)*item), 1 ) for item in [0,0.25,0.5,0.75,1]])
        
        try: fig.savefig( os.path.join(options['outdir'],options['logfname'].split('.')[0]+'_snr'+str(SNR_threshold)+'_hist.pdf'), bbox_inches='tight') #some matplotlib versions crash when saving pdf...
        except: fig.savefig( os.path.join(options['outdir'],options['logfname'].split('.')[0]+'_snr'+str(SNR_threshold)+'_hist.jpg'), bbox_inches='tight') #some matplotlib versions need pillow for jpg (conda install pillow)...   
        plt.close(fig)
    
    
    
    ###############################################################################
    #::: pyplot histograms (total counts)
    ###############################################################################
    #fig, ax = plt.subplots(figsize=(6.5,5))
    #h1,x,y,im = plt.hist2d(period[found==1], rplanet[found==1], bins=bins, cmap='Blues_r')
    #plt.colorbar(im, label='Recovery rate (%)')
    #plt.xlabel('Injected period (days)')
    #plt.ylabel(r'Injected radius (R$_\oplus$)')
    #change_font(ax)
    
    
    #fig, ax = plt.subplots(figsize=(6.5,5))
    #h2,x,y,im = plt.hist2d(period[found==0], rplanet[found==0], bins=bins, cmap='Blues_r')
    #plt.colorbar(im, label='Recovery rate (%)')
    #plt.xlabel('Injected period (days)')
    #plt.ylabel(r'Injected radius (R$_\oplus$)')
    #change_font(ax)
    
    
    
    
    ###############################################################################
    #::: kdeplots
    ###############################################################################
    # fig, ax = plt.subplots(figsize=(6.5,5))
    # ax = sns.kdeplot(period[found==0], rplanet[found==0], shade=True, cmap='Blues', cbar=True, alpha=0.5)
    # ax = sns.kdeplot(period[found==1], rplanet[found==1], shade=True, cmap='Blues_r', cbar=True)
    # ax.set(xlim=[0,120], ylim=[0.8,4])
    # plt.xlabel('Injected period (days)')
    # plt.ylabel(r'Injected radius (R$_\oplus$)')
    # change_font(ax)
    
    
    #fig, ax = plt.subplots(figsize=(6.5,5))
    #ax = sns.kdeplot(period[found==0], rplanet[found==0], shade=True, cmap='Blues', cbar=True)
    #ax.set(xlim=[15,85], ylim=[0.8,2.0])
    #plt.xlabel('Injected period (days)')
    #plt.ylabel(r'Injected radius (R$_\oplus$)')
    #change_font(ax)
    
    
    
    
    ###############################################################################
    #::: others
    ###############################################################################
    # plt.figure(figsize=(5,5))
    # z = found.reshape(len(np.unique(period)), len(np.unique(rplanet)))
    # plt.imshow(z.T, origin='lower', extent=(np.amin(period), np.amax(period), np.amin(rplanet), np.amax(rplanet)), aspect='auto', interpolation='gaussian', filterrad=5, cmap='Blues_r')
    # plt.xlabel('Period (days)')
    # plt.ylabel(r'Radius (R$_\oplus$)')
    
    #fig, ax = plt.subplots(figsize=(6.5,5))
    #plt.tricontourf(period, rplanet, found, cmap='Blues_r')
    #plt.xlabel('Injected period (days)')
    #plt.ylabel(r'Injected radius (R$_\oplus$)')
    
    
    # grid_x, grid_y = np.mgrid[np.amin(period):np.amax(period):100j, np.amin(rplanet):np.amax(rplanet):100j]
    # grid_z = griddata((period, rplanet), found*100, (grid_x, grid_y), method='linear')
    # fig, ax = plt.subplots(figsize=(6.5,5))
    # im = plt.imshow(grid_z.T, origin='lower', extent=(np.amin(period), np.amax(period), np.amin(rplanet), np.amax(rplanet)), interpolation='none', aspect='auto', cmap='Blues_r', rasterized=True, vmin=0, vmax=100)
    # plt.colorbar(im, label='Recovery rate (%)')
    # plt.xlabel('Injected period (days)')
    # plt.ylabel(r'Injected radius (R$_\oplus$)')
    #change_font(ax)
    #    
    #plt.savefig('injected_transit_search.pdf', bbox_inches='tight')       




###############################################################################
#::: run
###############################################################################
# plot('TIC_269701147.csv')




