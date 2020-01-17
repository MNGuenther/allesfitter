============================================
Tutorials 01b: Behind the GUI
============================================

The following series of tutorials picks up where the crash course left you, answers more questions and unlocks all advanced options. 

.. note:: We recommend using the GUI only as the very first step. It is designed to be filled in a linear sequence, step by step building up on your inputs. Once you went through all GUI steps, **do not go backwards and change things within the GUI**. Tuning selected settings or parameters is much easier by editing the created files manually in your text editor.


Setting things up
------------------------------------------------------------------------------
*allesfitter* expects a folder with 

- data files (e.g. `Leonardo.csv`)
- settings file (`settings.csv`)  
- parameters file (`params.csv`)
- stellar parameters file (`params_star.csv`; optional)

The GUI creates those for you in one go, but you can always amend these yourself in a text editor afterwards. The GUI hides all this from you, so let's have a look behind the scenes.



Example setup
------------------------------------------------------------------------------

Open the `tutorials/01_crash_course <https://github.com/MNGuenther/allesfitter/tree/master/tutorials/01_crash_course>`_ folder. You will see the following (folder and script names are arbitrary):

- a folder `allesfit`

  - data files (e.g. `Leonardo.csv`)
  - settings file (`settings.csv`) 
  - parameters file (`params.csv`)
  - stellar parameters file (`params_star.csv`; optional)

- a file `run.py`
- a file `simulate_data.py` (this was just used to simulate data and be ignored)



Data files
------------------------------------------------------------------------------

The folder `allesfit` is an example working directory for fitting one simple data set: `Leonardo.csv` (discovery photometry). Time, flux and flux error are given in a comma-separated .csv file. The file name has to match the telescope name.



settings.csv
------------------------------------------------------------------------------

Open it, and you will see that its minimal content are the planet letter ("b") and instrument ("Leonardo"). All of these must match the entries in `params.csv` and the names of any data .csv, here we only have `Leonardo.csv` files. To speed up the example, we also set the "fast_fit" option and run on "all" cores. There are dozens of other possible settings to give the user all freedom (see the following tutorials).



params.csv
------------------------------------------------------------------------------

Open it, and you will see the parameters describing the model. There are dozens of possible parameters to freeze and fit (see the following tutorials).



params_star.csv
------------------------------------------------------------------------------

This will be introduced in the next tutorials. This file is optional and, if given, contains the stellar parameters to calculate some derived parameters (e.g. the planet radius or equilibrium temperature).



run.py
------------------------------------------------------------------------------

This file just contains the simple 2-4 lines you need to execute to run any allesfit. For example, using Nested Sampling::

    import allesfitter
    allesfitter.show_initial_guess('allesfit')
    allesfitter.ns_fit('allesfit')
    allesfitter.ns_output('allesfit')

Or, if you're the MCMC kind of person::

    import allesfitter
    allesfitter.show_initial_guess('allesfit')
    allesfitter.mcmc_fit('allesfit')
    allesfitter.mcmc_output('allesfit')



Results folder
------------------------------------------------------------------------------

*allesfitter* creates the subfolder `results`, containing log files, result tables, LaTex tables, plots of the inital guess and final fit, corner plots, trace plots (for Nested Sampling), chain plots (for MCMC) and an internal save file. Have a look!


When you're ready, let's move onto the next tutorials.
