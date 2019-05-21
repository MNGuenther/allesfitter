============================================
Tutorials 01b: Behind the GUI
============================================

The following tutorials pick up where the crash course left you, answers a couple more questions, gives a peak behind the GUI, and unlocks some more advanced options. 

You have seen how the GUI can be launched via the `launch_allesfitter` app. 
Alternatively, you can also launch the GUI from a Python console::

    import allesfitter
    allesfitter.GUI()

But now let's actually take a look behind the GUI. Here, we explain how to allesfit your discovery transit using the python console.


Setting things up
------------------------------------------------------------------------------
*allesfitter* expects a folder with 

- all your data files (`Leonardo.csv`), 
- a settings file (`settings.csv`), and 
- a parameters file (`params.csv`). 

Then you let *allesfitter* run on that directory, and it does the rest. The GUI hides all this from you, so let's have a look behind the scenes. Open the `tutorials/01_crash_course
<https://github.com/MNGuenther/allesfitter/tree/master/tutorials/01_crash_course>`_ folder. You will see the following:

- a folder `allesfit` 
- a file `run.py`
- a file `simulate_data.py` (this was just used to simulate data and be ignored)



The `run.py` file
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



The working directory
------------------------------------------------------------------------------

The folder `allesfit` is an example working directory for fitting one simple data set: `Leonardo.csv` (discovery photometry). Time, flux and flux error are given in a comma-separated .csv file. The file name has to match the telescope name.

`settings.csv`. Open it, and you will see that its minimal content are the planet letter ("b") and instrument ("Leonardo"). All of these must match the entries in `params.csv` and the names of any data .csv, here we only have `Leonardo.csv` files. To speed up the example, we also set the "fast_fit" option and run on "all" cores. There are dozens of other possible settings to give the user all freedom (see the following tutorials).

`params.csv`. Open it, and you will see the parameters describing the model. There are dozens of possible parameters to freeze and fit (see the following tutorials).

Finally, when *allesfitter* runs, it creates the subfolder `results`. It contains log files, result tables, LaTex tables, plots of the inital guess and final fit, corner plots, trace plots (for Nested Sampling), chain plots (for MCMC) and an internal save file. Have a look!



Becoming advanced: 
------------------------------------------------------------------------------

Brilliant! You got great transit parameters that allowed to schedule follow-up observations, and you are now the hero of your team! They even baked you a cake! And even better: they nicely prepared all the old and new data for you in tutorial 04_transits_and_rvs:

* Leonardo.csv (discovery photometry)
* Michelangelo.csv (follow-up photometry)
* Donatello.csv (decent RV data)  
* Raphael.csv (good RV data)

Now let's allesfit all that data! Explore the `settings.csv` and `params.csv` files to see how all this additional data is included into a global fit for TMNT-1b. And/or, move on to the next tutorials page and learn how to tackle other data sets from scratch!
