#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:21:45 2020

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
try:
    import emcee
except:
    pass




#::: translate the mcmc_move string into a list of emcee commands
def get_move(x):
    '''
    Parameters
    ----------
    x : str
        Denoting a single mcmc_move, e.g., 'StretchMove'.

    Raises
    ------
    ValueError
        Alerts the user if the input was corrupted.

    Returns
    -------
    emcee.moves objects
        The emcee.moves object responding to the input string.
    '''
    if x == 'RedBlueMove': 
        return emcee.moves.RedBlueMove()
    elif x == 'StretchMove': 
        return emcee.moves.StretchMove()
    elif x == 'WalkMove': 
        return emcee.moves.WalkMove()
    elif x == 'KDEMove': 
        return emcee.moves.KDEMove()
    elif x == 'DEMove': 
        return emcee.moves.DEMove()
    elif x == 'DESnookerMove': 
        return emcee.moves.DESnookerMove()
    elif x == 'MHMove': 
        return emcee.moves.MHMove()
    elif x == 'GaussianMove': 
        return emcee.moves.GaussianMove()
    else:
        raise ValueError('Acceptable values for the setting mcmc_move are: '+\
                         'RedBlueMove / StretchMove / WalkMove / KDEMove /'+\
                         'DEMove / DESnookerMove / MHMove / GaussianMove.'+\
                         'You may also give mixtures, e.g.,'+\
                         '"DEMove 0.8 DESnookerMove 0.2".'+\
                         'Here, however, "'+x+'" was given.')
            


def translate_str_to_move(mcmc_move_str):
    '''
    Converts 'mcmc_move' user input into a list of objects that emcee understands.

    Parameters
    ----------
    mcmc_move_str : string
        What the user inputs into the alelsfitter settings.csv file.
        E.g., 'StretchMove' 
        E.g., 'DEMove 0.8 DESnookerMove 0.2'

    Returns
    -------
    command_list : list
        What emcee expects for its 'move' argument.
        E.g., [(<emcee.moves.stretch.StretchMove(), 1.0)]
        E.g., [(<emcee.moves.de.DEMove(), 0.8), (<emcee.moves.de_snooker.DESnookerMove(), 0.2)]
    '''
    mcmc_move_list = mcmc_move_str.split(' ')
    if len(mcmc_move_list)%2!=0: 
        mcmc_move_list.append('1') # in case the user gives only one move (and thus no weights)
    object_list = [get_move(x) for x in mcmc_move_list[::2]]
    weight_list = [float(x) for x in mcmc_move_list[1::2]]
    command_list = [(x,y) for x,y in zip(object_list,weight_list)]
    return command_list
    

        
if __name__ == '__main__':
    '''
    For testing only.
    '''
    translate_str_to_move('StretchMove')
    translate_str_to_move('StretchMove 1')
    translate_str_to_move('DEMove 0.8 DESnookerMove 0.2')
    