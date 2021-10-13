#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:37:00 2020

@author: 
Dr. Maximilian N. Günther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
GitHub: https://github.com/MNGuenther
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import ellc
from astropy.constants import G
from astropy import units as u

#::: my modules
from allesfitter.exoworlds_rdx.lightcurves import lightcurve_tools as lct

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})





###############################################################################
#::: translate physical parameters to ellc parameters
###############################################################################
def translate_physical_to_ellc(period = 1., 
                               R_companion = 1., M_companion = 1.,
                               R_companion_unit = 'Rearth', M_companion_unit = 'Mearth',
                               R_host = 1., M_host = 1., 
                               incl = 90, 
                               ecc = 0,
                               omega = 0):
    
    if R_companion_unit=='Rearth':
        R_companion_with_unit = R_companion*u.Rearth
    elif R_companion_unit=='Rjup':
        R_companion_with_unit = R_companion*u.Rjup
    elif R_companion_unit=='Rsun':
        R_companion_with_unit = R_companion*u.Rsun
        
    if M_companion_unit=='Mearth':
        M_companion_with_unit = M_companion*u.Mearth
    elif M_companion_unit=='Mjup':
        M_companion_with_unit = M_companion*u.Mjup
    elif R_companion_unit=='Msun':
        R_companion_with_unit = R_companion*u.Msun
    
    a = (G/(4*np.pi**2) * (period*u.d)**2 * (M_host*u.Msun + M_companion_with_unit))**(1./3.) #in AU, with astropy units
    a = a.to(u.AU) #in AU, with astropy units
    
    cosi = np.cos(np.deg2rad(incl))
    
    f_c = np.sqrt(ecc) * np.cos(np.deg2rad(omega))
    f_s = np.sqrt(ecc) * np.sin(np.deg2rad(omega))
    
    radius_1 = (R_host*u.Rsun / a).decompose().value
    radius_2 = (R_companion_with_unit / a).decompose().value
    a = a.value    

    return radius_1, radius_2, cosi, a, f_c, f_s



###############################################################################
#::: translate physical parameters to ellc parameters
###############################################################################
def translate_physical_to_allesfitter(period = 1., 
                                      R_companion = 1., M_companion = 1.,
                                      R_companion_unit = 'Rearth', M_companion_unit = 'Mearth',
                                      R_host = 1., M_host = 1., 
                                      incl = 90, 
                                      ecc = 0,
                                      omega = 0,
                                      ldc = [0.6,0.2],
                                      ld = 'quad'):
    
    if R_companion_unit=='Rearth':
        R_companion_with_unit = R_companion*u.Rearth
    elif R_companion_unit=='Rjup':
        R_companion_with_unit = R_companion*u.Rjup
    elif R_companion_unit=='Rsun':
        R_companion_with_unit = R_companion*u.Rsun
        
    if M_companion_unit=='Mearth':
        M_companion_with_unit = M_companion*u.Mearth
    elif M_companion_unit=='Mjup':
        M_companion_with_unit = M_companion*u.Mjup
    elif R_companion_unit=='Msun':
        R_companion_with_unit = R_companion*u.Msun
    
    cosi = np.cos(np.deg2rad(incl))
    
    a = (G/(4*np.pi**2) * (period*u.d)**2 * (M_host*u.Msun + M_companion_with_unit))**(1./3.) #in AU, with astropy units
    a = a.to(u.AU) #in AU, with astropy units
    
    f_c = np.sqrt(ecc) * np.cos(np.deg2rad(omega))
    f_s = np.sqrt(ecc) * np.sin(np.deg2rad(omega))
    
    if ld == 'quad':
        q1 = (ldc[0] + ldc[1])**2
        q2 = 0.5 * ldc[0] / (ldc[0] + ldc[1])
        ldc_transformed = [q1, q2]
        
    rr = ( R_companion_with_unit / (R_host*u.Rsun) ).decompose().value
    rsuma = ( (R_host*u.Rsun + R_companion_with_unit) / a ).decompose().value
    a = a.value    

    return rr, rsuma, cosi, a, f_c, f_s, ldc_transformed



###############################################################################
#::: make an ellc model lightcurve
###############################################################################
def make_lc_model(time, 
                  flux = None, flux_err = None,
                  epoch = 0., period = 1., 
                  R_companion = 1., M_companion = 1.,
                  R_companion_unit = 'Rearth', M_companion_unit = 'Mearth',
                  R_host = 1., M_host = 1., 
                  incl = 90, 
                  ecc = 0,
                  omega = 0,
                  ldc = [0.6,0.2],
                  ld = 'quad',
                  dil = 0,
                  sbratio = 0,
                  show_plot = False, save_plot = False, save_csv = False,
                  fname_plot = 'lc.pdf', fname_csv = 'lc.csv'):
    '''
    Inputs:
    -------
    time : array of float
        time in days
        
    Optional Inputs:
    ----------------
    flux : array of float
        flux of the 'underlying' lightcurve
    flux_err : array of float
        flux error of the 'underlying' lightcurve
    epoch : float
        epoch in days
    period : float
        period in days
    R_companion: float
        radius of the companion
        default is 1 Rearth
    M_companion: float
        mass of the companion
        default is 1 Mearth
    R_companion: float
        radius of the companion
        default is 1 Rearth
    M_companion: float
        mass of the companion
        default is 1 Mearth
    R_host : float
        radius of the star, in Rsun
        default is 1
    M_host: float
        mass of the star, in Msun
        default is 1
    show_plot : bool
        show the plot in the terminal, or close it
        default is False
    save_plot : bool
        save the plot to a file, or not
        default is False
    save_csv : bool
        save the lightcurve to a file, or not
        default is False
        
    Returns:
    --------
    model_flux : array of float
        relative flux of the model
    '''
    
    radius_1, radius_2, cosi, a, f_c, f_s = translate_physical_to_ellc(period = period, R_companion = R_companion, M_companion = M_companion, R_companion_unit = R_companion_unit, M_companion_unit = M_companion_unit, R_host = R_host, M_host = M_host, incl = incl, ecc = ecc, omega = omega)   
    rr, rsuma, cosi, a, f_c, f_s, ldc_transformed = translate_physical_to_allesfitter(period = period, R_companion = R_companion, M_companion = M_companion, R_companion_unit = R_companion_unit, M_companion_unit = M_companion_unit, R_host = R_host, M_host = M_host, incl = incl, ecc = ecc, omega = omega)   

    model_flux = ellc.lc(t_obs = time, radius_1 = radius_1, radius_2 = radius_2, sbratio = sbratio, incl = incl, light_3 = dil, t_zero = epoch, period = period, a = a, q = 1, f_c = f_c, f_s = f_s, ldc_1=ldc, ldc_2 = None, gdc_1 = None, gdc_2 = None, didt = None, domdt = None, rotfac_1 = 1, rotfac_2 = 1, hf_1 = 1.5, hf_2 = 1.5, bfac_1 = None, bfac_2 = None, heat_1 = None, heat_2 = None, lambda_1 = None, lambda_2 = None, vsini_1 = None, vsini_2 = None, t_exp=None, n_int=None,  grid_1='default', grid_2='default', ld_1=ld, ld_2=None, shape_1='sphere', shape_2='sphere', spots_1=None, spots_2=None, exact_grav=False, verbose=1)
    
    if flux is not None:
        flux2 = flux+model_flux-1
    else:
        flux2 = model_flux
        
    if show_plot or save_plot:        
        
        #::: get phase-folded data
        phase, phaseflux, phaseflux_err, N, phi = lct.phase_fold(time, flux2, period, epoch, dt=0.002, ferr_type='medsig', ferr_style='sem', sigmaclip=True)

        #::: get model on fine grid from phase -0.25 to 0.75
        xx = np.linspace( epoch-0.25*period, epoch+0.75*period, 1000 )
        xx_phase = np.linspace(-0.25, 0.75, 1000 )
        model_flux_xx_phase = ellc.lc(t_obs = xx, radius_1 = radius_1, radius_2 = radius_2, sbratio = sbratio, incl = incl, light_3 = dil, t_zero = epoch, period = period, a = a, q = 1, f_c = f_c, f_s = f_s, ldc_1=ldc, ldc_2 = None, gdc_1 = None, gdc_2 = None, didt = None, domdt = None, rotfac_1 = 1, rotfac_2 = 1, hf_1 = 1.5, hf_2 = 1.5, bfac_1 = None, bfac_2 = None, heat_1 = None, heat_2 = None, lambda_1 = None, lambda_2 = None, vsini_1 = None, vsini_2 = None, t_exp=None, n_int=None,  grid_1='default', grid_2='default', ld_1=ld, ld_2=None, shape_1='sphere', shape_2='sphere', spots_1=None, spots_2=None, exact_grav=False, verbose=1)
            
        fig = plt.figure(figsize=(15,10), tight_layout=True)
        gs = gridspec.GridSpec(3, 3)
        
        #::: full curve
        ax = fig.add_subplot(gs[0,:])
        ax.plot(time, flux2, 'b.', rasterized=True, zorder=10)
        ax.plot(time, model_flux, 'r-', lw=2, zorder=12)
        ax.set(ylabel='Flux', xlabel='Time'+r' $\mathrm{(BJD_{TDB})}$')
        
        #::: phase-folded curve
        ax = fig.add_subplot(gs[1,0])
        ax.plot(phi, flux2, 'k.', color='lightgrey', rasterized=True, zorder=10)
        ax.errorbar(phase, phaseflux, yerr=phaseflux_err, fmt='b.', zorder=11)
        ax.plot(xx_phase, model_flux_xx_phase, 'r-', lw=2, zorder=12)
        ax.set(xlabel='Phase', ylabel='Flux')
        
        #::: phase-folded and zoomed curve; same as for axes[0,1], but zoomed in x
        ax = fig.add_subplot(gs[1,1])
        ax.plot(phi*period*24.*60, flux2, 'k.', color='lightgrey', rasterized=True, zorder=10)
        ax.errorbar(phase*period*24.*60, phaseflux, yerr=phaseflux_err, fmt='b.', zorder=11)
        ax.plot(xx_phase*period*24.*60, model_flux_xx_phase, 'r-', lw=2, zorder=12)
        ax.set(xlim=[-0.03*period*24.*60,0.03*period*24.*60])
        ax.set(xlabel='Time (min.)')
        
        #::: info
        ax = fig.add_subplot(gs[2,0])
        ax.set_axis_off()
        ax.text(0,0.95,r'$T_0$' + ' = '+str(epoch)+r' $\mathrm{BJD_{TDB}}$', transform=ax.transAxes)
        ax.text(0,0.85,'P = '+str(period)+' days', transform=ax.transAxes)
        ax.text(0,0.75,'R_comp = '+str(R_companion)+' '+R_companion_unit, transform=ax.transAxes)
        ax.text(0,0.65,'M_comp = '+str(M_companion)+' '+M_companion_unit, transform=ax.transAxes)
        ax.text(0,0.55,'R_host = '+str(R_host)+' Rsun', transform=ax.transAxes)
        ax.text(0,0.45,'M_host = '+str(M_host)+' Msun', transform=ax.transAxes)
        ax.text(0,0.35,'incl = '+str(incl)+' deg', transform=ax.transAxes)
        ax.text(0,0.25,'ecc = '+str(ecc), transform=ax.transAxes)
        ax.text(0,0.15,'omega = '+str(omega)+' deg', transform=ax.transAxes)
        ax.text(0,0.05,'LD = '+str(ldc)+' '+ld, transform=ax.transAxes)
        
        ax = fig.add_subplot(gs[2,1])
        ax.set_axis_off()
        ax.text(0,0.95,'dil = '+str(dil), transform=ax.transAxes)
        ax.text(0,0.85,'sbratio = '+str(sbratio), transform=ax.transAxes)
        ax.text(0,0.75,'R_comp/R_host = '+np.format_float_positional(rr,5,False), transform=ax.transAxes)
        ax.text(0,0.65,'(R_comp+R_host)/a = '+np.format_float_positional(rsuma,5,False), transform=ax.transAxes)
        ax.text(0,0.55,'R_comp/a = '+np.format_float_positional(radius_2,5,False), transform=ax.transAxes)
        ax.text(0,0.45,'R_host/a = '+np.format_float_positional(radius_1,5,False), transform=ax.transAxes)
        ax.text(0,0.35,'cosi = '+np.format_float_positional(cosi,5,False), transform=ax.transAxes)
        ax.text(0,0.25,r'$\sqrt{e} \cos{\omega}$ = '+np.format_float_positional(f_c,5,False), transform=ax.transAxes)
        ax.text(0,0.15,r'$\sqrt{e} \sin{\omega}$ = '+np.format_float_positional(f_s,5,False), transform=ax.transAxes)
        ax.text(0,0.05,'LD transf = ['+", ".join([np.format_float_positional(item,5,False) for item in ldc_transformed]) + ']', transform=ax.transAxes)
        
        if show_plot:
            plt.show(fig)
            
        if save_plot:
            if len(os.path.dirname(fname_plot))>0 and not os.path.exists(os.path.dirname(fname_plot)): os.makedirs(os.path.dirname(fname_plot))
            fig.savefig(fname_plot)
            plt.close(fig)
    
    if save_csv:
        if len(os.path.dirname(fname_csv))>0 and not os.path.exists(os.path.dirname(fname_csv)): os.makedirs(os.path.dirname(fname_csv))
        X = np.column_stack((time, flux2, flux_err))
        np.savetxt(fname_csv, X, delimiter=',')


    return flux2

    
    
###############################################################################
#::: inject an ellc model lightcurve
###############################################################################
def inject_lc_model(time, flux, flux_err, **kwargs):
    '''
    Wrapper around make_lc_model()
    '''
    
    return make_lc_model(time, flux = flux, flux_err = flux_err, **kwargs)
    


   
###############################################################################
#::: make an ellc model lightcurve
###############################################################################
def inject_rv_model(time, rv, rv_err, 
               epoch, period, 
               R_companion= 1., M_companion=1., 
               R_companion_unit = 'Rearth', M_companion_unit = 'Mearth',
               R_host = 1., M_host = 1., 
               sbratio = 0,
               incl = 90, 
               ecc = 0,
               omega = 0,
               dil = 0,
               ldc = [0.6,0.2],
               ld = 'quad',
               show_plot = False, save_plot = False, save_csv = False,
               outdir = '', fname_plot=None, fname_csv=None):
    '''
    Inputs:
    -------
    time : array of float
        time in days
    rv : array of float
        RV of the 'underlying' series
    rv_err : array of float
        error of RV of the 'underlying' series
    epoch : float
        epoch in days
    period : float
        period in days
    R_companion: float
        radius of the planet, in R_earth
        
    Optional Inputs:
    ----------------
    M_companion: float
        mass of the planet, in M_earth
        default is 0
    R_host : float
        radius of the star, in R_sun
        default is 1
    M_host: float
        mass of the star, in M_sun
        default is 1
    show_plot : bool
        show the plot in the terminal, or close it
        default is False
    save_plot : bool
        save the plot to a file, or not
        default is False
    save_csv : bool
        save the lightcurve to a file, or not
        default is False
        
    Returns:
    --------
    flux2 : array of float
        relative flux with injected signal
    '''
    
    if R_companion_unit=='Rearth':
        R_companion_with_unit = R_companion*u.Rearth
    elif R_companion_unit=='Rjup':
        R_companion_with_unit = R_companion*u.Rjup
        
    if M_companion_unit=='Mearth':
        M_companion_with_unit = M_companion*u.Mearth
    elif M_companion_unit=='Mjup':
        M_companion_with_unit = M_companion*u.Mjup
    
    a = (G/(4*np.pi**2) * (period*u.d)**2 * (M_host*u.Msun + M_companion_with_unit))**(1./3.) #in AU, with astropy units
    a = a.to(u.AU) #in AU, with astropy units
    
    if ecc>0:
        f_c = np.sqrt(ecc) * np.cos(np.deg2rad(omega))
        f_s = np.sqrt(ecc) * np.sin(np.deg2rad(omega))
    else:
        f_c = None
        f_s = None
        
    radius_1 = (R_host*u.Rsun / a).decompose().value
    radius_2 = (R_companion_with_unit / a).decompose().value
    a = a.value
    
    def get_model_rv(xx):
        return ellc.rv(t_obs = xx, 
                       radius_1 = radius_1,
                       radius_2 = radius_2,
                       sbratio = sbratio, 
                       incl = incl, 
                       t_zero = epoch, 
                       period = period,
                       a = a,
                       q = 1,
                       f_c = f_c, f_s = f_s,
                       ldc_1=ldc, ldc_2 = None,
                       gdc_1 = None, gdc_2 = None,
                       didt = None,
                       domdt = None,
                       rotfac_1 = 1, rotfac_2 = 1,
                       hf_1 = 1.5, hf_2 = 1.5,
                       bfac_1 = None, bfac_2 = None,
                       heat_1 = None, heat_2 = None, 
                       lambda_1 = None, lambda_2 = None,
                       vsini_1 = None, vsini_2 = None,
                       t_exp=None, n_int=None, 
                       grid_1='default', grid_2='default',
                       ld_1=ld, ld_2=None,
                       shape_1='sphere', shape_2='sphere',
                       spots_1=None, spots_2=None, 
                       verbose=1)[0]
    
    #::: get model on data grid
    model_rv = get_model_rv(time)
    rv2 = rv+model_rv
    
    if show_plot or save_plot:
        
        #::: get phase 
        phi = lct.calc_phase(time, period, epoch)
        phi[phi>0.75] -= 1.

        #::: get model on fine grid
        xx1 = np.arange( time[0], time[-1], 0.1 )
        model_rv_xx1 = get_model_rv(xx1)
    
        #::: get model on fine grid from phase -0.25 to 0.75
        xx = np.linspace( epoch-0.25*period, epoch+0.75*period, 1000 )
        xx_phase = np.linspace(-0.25, 0.75, 1000 )
        model_rv_xx_phase = get_model_rv(xx)
    
        fig, axes = plt.subplots(1, 2, figsize=(16,5), sharey='row', gridspec_kw={'width_ratios': [3,1]})
        
        #::: full curve
        axes[0].errorbar(time, rv2, yerr=rv_err, fmt='bo', zorder=11)
        axes[0].plot(xx1, model_rv_xx1, 'r-', lw=2, zorder=10)
        axes[0].set(ylabel='RV (km/s)', xlabel='Time (BJD_TDB)')
        
        #::: phase-folded curve
        axes[1].errorbar(phi, rv2, yerr=rv_err, fmt='b.', zorder=11)
        axes[1].plot(xx_phase, model_rv_xx_phase, 'r-', lw=2, zorder=10)
        axes[1].set(xlabel='Phase')
        
        suptitle = fig.suptitle('\nP = '+str(period)+' days, Rp = '+str(R_companion)+' '+R_companion_unit, y=1.15)
        fig.tight_layout()        

        if show_plot:
            plt.show(fig)
            
        if save_plot:
            if fname_plot is None: 
                if not os.path.exists(os.path.join(outdir,'plots')): os.makedirs(os.path.join(outdir,'plots'))
                fname_plot = os.path.join(outdir,'plots','Lightcurve P = '+str(period)+' days, Rp = '+str(R_companion)+' '+R_companion_unit+'.pdf')
            fig.savefig(fname_plot, bbox_extra_artists=(suptitle,), bbox_inches="tight")
            plt.close(fig)
    
    if save_csv:
        if fname_csv is None: 
            if not os.path.exists(os.path.join(outdir,'csv')): os.makedirs(os.path.join(outdir,'csv'))
            fname_csv = os.path.join(outdir,'csv','Lightcurve P = '+str(period)+' days, Rp = '+str(R_companion)+' '+R_companion_unit+'.csv')
        X = np.column_stack((time, rv2, rv_err))
        np.savetxt(fname_csv, X, delimiter=',')


    return rv2