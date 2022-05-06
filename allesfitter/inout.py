#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:32:01 2020

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
import json

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})




def write_csv(fname, *arrays, **kwargs):
    '''
    Writes multiple arrays to a csv file, e.g., time, flux, flux_err.

    Parameters
    ----------
    fname : str
        Name of the file (and directory).
    *arrays : collection of arrays
        One or multiple arrays, e.g. time, flux, flux_err.
    **kwargs : collection of keyword arguments
        Any kwargs for np.savetxt(), e.g., fmt=['%.18e','%.12e','%.12e']

    Returns
    -------
    None.

    Outputs
    -------
    Saves a csv file under the given name.
    '''
    X = np.column_stack(arrays)
    np.savetxt(fname, X, delimiter=',', **kwargs)



def read_csv(fname, skip_header=0):
    '''
    Reads a csv file and unpacks the columns.

    Parameters
    ----------
    fname : str
        Name of the file (and directory).

    Returns
    -------
    Collection of arrays, e.g. time, flux, flux_err.
    '''
    return np.genfromtxt(fname, delimiter=',', comments='#', encoding='utf-8', dtype=float, unpack=True, skip_header=skip_header)



def write_json(fname, dic):
    '''
    Writes something to a json file, e.g. a dictionary.

    Parameters
    ----------
    fname : str
        Name of the file (and directory).
    dic : dictionary
        A dictionary (or other collection) to be saved.

    Returns
    -------
    None.

    Outputs
    -------
    Saves a json file under the given name.
    '''
    with open(fname, 'w') as fp:
        json.dump(dic, fp, indent=4)



def read_json(fname):
    '''
    Reads a json file and retrieves the content.

    Parameters
    ----------
    fname : str
        Name of the file (and directory).

    Returns
    -------
    A dictionary (or other collection).
    '''
    with open(fname, 'r') as fp:
        dic = json.load(fp)
    return dic