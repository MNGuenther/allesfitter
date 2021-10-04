#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:15:19 2020

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
from pprint import pprint

#::: optional modules
try:
    from wotan import flatten
except ImportError:
    pass

#::: my modules
from allesfitter import tessplot
from allesfitter.time_series import sigma_clip, slide_clip, mask_regions
from allesfitter.detection.periodicity import estimate_period
from allesfitter.detection.transit_search import get_tls_kwargs_by_tic, tls_search
from exoworlds.tess import tessio

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




#::: 0) tic_id
tic_id = '390651552'

#::: 1) load data
time, flux, flux_err = tessio.get(tic_id, unpack=True)
tessplot(time, flux, clip=False)
tessplot(time, flux)

#::: 2) prepare data
flux = sigma_clip(time, flux, high=3, low=20)
flux = slide_clip(time, flux, high=3, low=20)
flux = mask_regions(time, flux, regions=[(2458865.5,2458866), (2458990,2458990.5)])
tessplot(time, flux)

#::: 3) estimate variability period
period = estimate_period(time, flux, flux_err)[0]

#::: 4) detrend data
flux = slide_clip(time, flux, high=3, low=20)
flux_flat, trend = flatten(time, flux, method='biweight', window_length=1, return_trend=True)
tessplot(time, flux, trend=trend)
tessplot(time, flux_flat)
period = estimate_period(time, flux_flat, flux_err)[0]

#::: 5) search transits
kwargs = get_tls_kwargs_by_tic(tic_id) #390651552 #459837008
results_all, fig_all = tls_search(time, flux_flat, flux_err, plot=True, 
                                    period_min=1, period_max=20, **kwargs)
for r in results_all:
    print('---------------------')
    pprint(r)
    tessplot(time, flux_flat, trend=r['model'], clip=True)
