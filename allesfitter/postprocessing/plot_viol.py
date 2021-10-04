"""
Pipeline to analyze allesfitter output for planet transit timings

argument 1: allesfitter path
argument 2: p-value threshold
argument 3: Boolean to select to plot wout/with TESS or wout/with/only TESS

Tansu Daylan
MIT Kavli Institute, Cambridge, MA, 02109, US
tansu.daylan@gmail.com
www.tansudaylan.com
"""

import numpy as np
import scipy
import os, datetime, sys

import matplotlib.pyplot as plt
from tdpy.util import summgene 

import allesfitter
import allesfitter.postprocessing.plot_viol
from allesfitter import config

import astropy

class gdatstrt(object):

    def __init__(self):
        self.boollockmodi = False
        pass
    
    def __setattr__(self, attr, valu):
        super(gdatstrt, self).__setattr__(attr, valu)


def plot(gdat, indxstar, indxpara=None, strgtype='evol'):
    
    if indxstar.size == 1:
        strg = gdat.liststrgstar[indxstar[0]] + '_'
    else:
        strg = ''
    
    print('strgtype')
    print(strgtype)
    
    listticklabl = []
    if strgtype == 'epocevol':
        chanlist = [[[] for m in gdat.indxstar] for i in gdat.indxruns]
        xpos = np.array(gdat.listyear)
        for i in gdat.indxruns:
            for m in indxstar:
                chanlist[i][m] = [gdat.timejwst[k][i][m] for k in gdat.indxyear]
        for k in gdat.indxyear:
            listticklabl.append('%s' % str(gdat.listyear[k]))
    else:
        chanlist = []
        numbxdat = gdat.numbruns * indxstar.size
        xpos = 0.6 * (np.arange(numbxdat) + 1.)

        for i in gdat.indxruns:
            for m in indxstar:
                if strgtype == 'jwstcomp':
                    chanlist.append(gdat.timejwst[1][i][m])
                if strgtype == 'paracomp':
                    for k in indxpara:
                        chanlist.append((gdat.listobjtalle[i][m].posterior_params[gdat.liststrgparaconc[k]] - \
                            np.mean(gdat.listobjtalle[i][m].posterior_params[gdat.liststrgparaconc[k]])) * 24. * 60.)
                    
                if strgtype == 'paracomp' or strgtype == 'jwstcomp':
                    ticklabl = '%s, %s' % (gdat.liststrgstar[m], gdat.liststrgruns[i])
                    listticklabl.append(ticklabl)
                else:
                    ticklabl = '%s, %s' % (gdat.liststrgstar[m], gdat.liststrgruns[i])
                    listticklabl.append(ticklabl)
                
        if xpos.size != len(listticklabl):
            raise Exception('')
        
    print('xpos')
    summgene(xpos)
    print('chanlist')
    print(chanlist)
    figr, axis = plt.subplots(figsize=(5, 4))
    if strgtype != 'epocevol':
        axis.violinplot(chanlist, xpos, showmedians=True, showextrema=False)
    else:
        for i in gdat.indxruns:
            for m in indxstar:
                axis.violinplot(chanlist[i][m], xpos, showmedians=True, showextrema=False)

    axis.set_xticks(xpos)
    if strgtype == 'jwstcomp':
        axis.set_ylabel('Transit time residual in 2023 [min]')
        strgbase = strgtype

    if strgtype == 'paracomp':
        if gdat.liststrgparaconc[indxpara] == 'b_period':
            axis.set_ylabel('P [min]')
        else:
            labl = gdat.listlablparaconc[indxpara[0]]
            axis.set_ylabel(labl)
        strgbase = '%04d' % indxpara
    
    if strgtype == 'epocevol':
        axis.set_xlabel('Year')
        axis.set_ylabel('Transit time residual [min]')
        strgbase = strgtype
    
    path = gdat.pathimag + 'viol_%s.%s' % (strgbase, gdat.strgplotextn)
    axis.set_xticklabels(listticklabl)
    plt.tight_layout()
    print('Writing to %s...' % path)
    print()
    figr.savefig(path)
    plt.close()
    

def plot_viol(pathbase, liststrgstar, liststrgruns, lablstrgruns, pathimag, pvalthrs=1e-3):

    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('allesfitter postprocessing violin plot started at %s...' % strgtimestmp)
    
    # construct global object
    gdat = gdatstrt()
    
    # copy unnamed inputs to the global object
    #for attr, valu in locals().iter():
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # runs to be compared for each star
    gdat.numbruns = len(liststrgruns)
    gdat.indxruns = np.arange(gdat.numbruns)
    
    gdat.pathimag = pathimag
    gdat.liststrgstar = liststrgstar

    # stars
    numbstar = len(liststrgstar)
    gdat.indxstar = np.arange(numbstar)

    # plotting
    gdat.strgplotextn = 'png'

    # read parameter keys, labels and posterior from allesfitter output
    liststrgpara = [[] for i in gdat.indxruns]
    listlablpara = [[] for i in gdat.indxruns]
    gdat.listobjtalle = [[[] for m in gdat.indxstar] for i in gdat.indxruns]
    for i in gdat.indxruns:
        for m in gdat.indxstar:
            pathalle = pathbase + '%s/allesfits/allesfit_%s/' % (gdat.liststrgstar[m], gdat.liststrgruns[i])
            print('Reading from %s...' % pathalle)
            config.init(pathalle)
            liststrgpara[i] = np.array(config.BASEMENT.fitkeys)
            listlablpara[i] = np.array(config.BASEMENT.fitlabels)
            # read the chain
            print('pathalle')
            print(pathalle)
            gdat.listobjtalle[i][m] = allesfitter.allesclass(pathalle)
    
    # concatenate the keys, labels from different runs
    gdat.liststrgparaconc = np.concatenate(liststrgpara)
    gdat.liststrgparaconc = np.unique(gdat.liststrgparaconc)
    gdat.listlablparaconc = np.copy(gdat.liststrgparaconc)
    for k, strgparaconc in enumerate(gdat.liststrgparaconc):
        for i, strgruns in enumerate(liststrgruns):
            if strgparaconc in liststrgpara[i]:
                gdat.listlablparaconc[k] = listlablpara[i][np.where(liststrgpara[i] == strgparaconc)[0][0]]
    
    gdat.numbparaconc = len(gdat.liststrgparaconc)
    gdat.indxparaconc = np.arange(gdat.numbparaconc)
    for k, strgpara in enumerate(gdat.liststrgparaconc):
        booltemp = True
        for i in gdat.indxruns:
            if not strgpara in liststrgpara[i]:
                booltemp = False
        if not booltemp:
            continue
        
        ## violin plot
        ## mid-transit time prediction
        plot(gdat, gdat.indxstar, indxpara=np.array([k]), strgtype='paracomp')
        ## per-star 
        #for m in gdat.indxstar:
        #    plot(gdat, indxstar=np.array([m]), indxpara=k, strgtype='paracomp')
        
    # calculate the future evolution of epoch
    gdat.listyear = [2021, 2023, 2025]
    numbyear = len(gdat.listyear)
    gdat.indxyear = np.arange(numbyear)
    gdat.timejwst = [[[[] for m in gdat.indxstar] for i in gdat.indxruns] for k in gdat.indxyear]
    for k, year in enumerate(gdat.listyear):
        epocjwst = astropy.time.Time('%d-01-01 00:00:00' % year, format='iso').jd
        for i in gdat.indxruns:
            for m in gdat.indxstar:
                epoc = gdat.listobjtalle[i][m].posterior_params['b_epoch']
                peri = gdat.listobjtalle[i][m].posterior_params['b_period']
                indxtran = (epocjwst - epoc) / peri
                indxtran = np.mean(np.rint(indxtran))
                if indxtran.size != np.unique(indxtran).size:
                    raise Exception('')
                gdat.timejwst[k][i][m] = epoc + peri * indxtran
                gdat.timejwst[k][i][m] -= np.mean(gdat.timejwst[k][i][m])
                gdat.timejwst[k][i][m] *= 24. * 60.
    
    listfigr = []
    listaxis = []

    # temporal evolution of mid-transit time prediction
    plot(gdat, gdat.indxstar, strgtype='epocevol')
    ## per-star 
    #for m in gdat.indxstar:
    #    plot(gdat, indxstar=np.array([m]), strgtype='epocevol')
    
    ## mid-transit time prediction
    plot(gdat, gdat.indxstar, strgtype='jwstcomp')
    ## per-star 
    #for m in gdat.indxstar:
    #    plot(gdat, indxstar=np.array([m]), strgtype='jwstcomp')
    
    return listfigr, listaxis


    


