=======================================
allesfitter
=======================================

[German-ish for *everything-fitter*]

[beta version]

*allesfitter* is a convenient wrapper around the packages *ellc* (planet and EB light curve and RV models), *dynesty* (static and dynamic nested sampling) *emcee* (Markov Chain Monte Carlo sampling) and *celerite* (Gaussian Process models).

*allesfitter* is suited for the analysis of multi-planet systems observed with various photometric and RV instruments. It is highly user-friendly, flexible, robust, and allows a wide choice of sampling algorithms and baseline detrending. The user defines all input parameters and settings in a text file, and *allesfitter* automatically runs the nested sampling or MCMC fit, and produces all output such as tables, latex tables, and plots. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



# allesfitter

[German-ish for *everything-fitter*]

[beta version]

*allesfitter* is a convenient wrapper around the packages *ellc* (planet and EB light curve and RV models), *dynesty* (static and dynamic nested sampling) *emcee* (Markov Chain Monte Carlo sampling) and *celerite* (Gaussian Process models).

*allesfitter* is suited for the analysis of multi-planet systems observed with various photometric and RV instruments. It is highly user-friendly, flexible, robust, and allows a wide choice of sampling algorithms and baseline detrending. The user defines all input parameters and settings in a text file, and *allesfitter* automatically runs the nested sampling or MCMC fit, and produces all output such as tables, latex tables, and plots. 


## Table of contents
1. Acknowledge / cite
2. Why *allesfitter*?
3. Crash course
    3.1 Using the GUI
    3.2 Using the python console
4. Installation
5. Tutorial 
   5.1 Setting things up
   5.2 Running the fit
6. Examples
7. Performance & timing
References



## 1. Acknowledge & cite

**Contributors:** Maximilian N. Günther, Tansu Daylan

**License:** The code is freely available at https://github.com/MNGuenther/allesfitter under the MIT License. Feedback and contributions are very welcome.

**Cite:** If you use *allesfitter* or parts of it in your work, please cite *Günther et al., in prep.* (link). Please also include the following acknowledgement: "This work makes use of the *allesfitter* package (*Günther et al., in prep.*), which is a convenient wrapper around the packages *ellc* (Maxted 2016), *aflare1.py* (Davenport 2014), *dynesty* (https://github.com/joshspeagle/dynesty), *emcee* (Foreman-Mackey 2013) and *celerite* (Foreman-Mackey 2017). This work makes further use of the *python* programming language (Rossum 1995) and the open-source *python* packages *numpy* (van der Walt, Colbert & Varoquaux 2011), *scipy* (Jones et al. 2001), *matplotlib* (Hunter 2007), *tqdm* (doi:10.5281/zenodo.1468033) and *seaborn* (https://seaborn.pydata.org/index.html)."


## 2. Why allesfitter?
- GUI, easy-to-use & all-in-one
- various MCMC and Nested Sampling* algorithms (static vs. dynamic, multinest, slicing, ...)**
- exoplanets & eclipsing binaries.***
- model all your photometric and radial velocity data jointly
- various baseline and noise fitting options (sampling vs. hybrid, GPs, splines, polynomials, ...)
- click a button, get a coffee, and let *allesfitter* write your paper (it creates all latex tables and plots).

*get all that Bayesian evidence and do a meaningful model comparison! Is there evidence for an occultation? Do you see phase-variations or systematic noise patterns? Are those TTVs meaningful? Is your orbit circular or eccentric? So many questions - so much Bayesian evidence!
**powered by emcee (Foreman-Mackey, 2013) and dynesty (Speagal, 2018)
***powered by ellc (Maxted, 2016)


## 3. Installation
For now...

- git clone https://github.com/MNGuenther/allesfitter into your PYTHONPATH.
- git clone https://github.com/MNGuenther/exoworlds into your PYTHONPATH.
- (soon to be pip-installable)
 
Also...

- pip install ellc (>=1.8.0)
	- (this requires that a Fortran compiler is installed. If it is missing, you can use e.g. homebrew with "brew install gcc" to install one.)
- pip install corner (>=2.0.1)
- pip install emcee (>=3.0.0) 
	- (if you want to run MCMCs)
- git clone https://github.com/joshspeagle/dynesty.git (>=0.9.2b) into your PYTHONPATH 
	- (if you want to run Nested Sampling)
- pip install celerite (>=0.3.0 )
	- (if you want to run Gaussian Processes)


## 4. Crash course - TMNT found its first planet!

Imagine your survey called "TMNT" found a planet using its photometric telescope "Leonardo". Now you want to model it, so you can schedule follow-up with the rest of your TMNT network: Michelangelo (photometry), Donatello (RV) and Raphael (RV). 

### 4.1 Using the GUI 
Launch the GUI:

    import allesfitter
    allesfitter.GUI()

Fill out the fields step by step, as demonstrated in this YouTube tutorial (link).



### 4.2 Using the python console 

The way *allesfitter* works is that you prepare a folder with all your data files (`Leonardo.csv`), a settings file (`settings.csv`) and a parameters file (`params.csv`). Then you let *allesfitter* run on that directory, and it does the rest. The GUI hides all this from you, so let's have a look behind the scenes.

Open the `examples/crash_course` folder. You will see the file `run.py` and the following two folders: `allesfit_Leonardo` and `allesfit_all_TMNT`.

### `run.py`
This file just contains the simple 3 lines you need to execute to run any of the examples below (after installation). For example, using Nested Sampling:

    import allesfitter
    allesfitter.ns_fit('allesfit_Leonardo')
    allesfitter.ns_output('allesfit_Leonardo')

Or, if you're the MCMC kind of person:

    import allesfitter
    allesfitter.mcmc_fit('allesfit_Leonardo')
    allesfitter.mcmc_output('allesfit_Leonardo')
    
### `allesfit_Leonardo`: 

This folder is an example of fitting the following data set: `Leonardo.csv` (discovery photometry). Time, flux and flux error are given in a comma-separated .csv file.

`settings.csv`. Open it, and you will see that its minimal content are the planet letter ("b") and instrument ("Leonardo"). All of these must match the entries in `params.csv` and the names of any data .csv, here we only have `Leonardo.csv` files. To speed up the example, we also set the fast_fit option and run on 4 cores. There are dozens of other possible settings to give the user all freedom. These are explained further in Section X below. (todo)

`params.csv`. Open it, and you will see the parameters describing the model. There are dozens of possible parameters to freeze and fit. These are further explained in Section Y below. (todo)

Finally, when *allesfitter* runs, it creates the subfolder `results`. It contains log files, result tables, LaTex tables, plots of the inital guess and final fit, corner plots, trace plots (for Nested Sampling), chain plots (for MCMC) and an internal save file. Have a look!


### `allesfit_all_TMNT`: 

This is an example of fitting the following four data sets:

  - Leonardo.csv (discovery photometry)
  - Michelangelo.csv (follow-up photometry)
  - Donatello.csv (decent RV data)
  - Raphael.csv (good RV data)

Explore the `settings.csv` and `params.csv` files to see how to include this additional data into the fit.




## 5. Tutorial

### 5.1 Settings things up
The file structure is as follows:
- datadir
  * settings.csv
  * params.csv
  * tel1.csv
  * tel2.csv
  * ...		

#### datadir
For example, datadir = "users/johnwayne/tess_1b".

#### settings.csv
- copy settings.csv from the "examples" that come with allesfitter
- adjust as you need
(todo)

#### params.csv
- copy params.csv from the "examples" that come with allesfitter
- adjust as you need
(todo)

#### tel1.csv, tel2.csv, ...
All these telescope data files must be named exactly like the telesopes listed in settings.csv. So for example "TESS.csv", "HARPS.csv", and so on. The code will automatically search for these file names in the given directory.

Photometric instruments must contain three columns:
	time	flux	flux_err

RV instruments must contain three columns:
	time	rv	rv_err
  
The time unit has to be days, preferably BJD. It does not matter if it is JD/HJD/BJD, but it *must* be homogenous between all data sets tel1.csv etc. and the parameters (epoch) in params.csv.


### 5.2 Running the fit
- copy run.py from the "examples" that come with allesfitter
- adjust as you need
(todo)


## 6. Examples

### `examples/crash_course/`

### `examples/TMNT/`
Various different fitting scenarios and different settings for the TMNT data set.

### `examples/simulate_data/`
How the TMNT data set was simulated.



## 7. Performance & timining

### `examples/TMNT/`

#### a)
Fitting the Leonardo discovery photometry using Dynamic Nested Sampling, once with uniform sampling, once with random-walk sampling:

    ns_modus,dynamic
    ns_nlive,500
    ns_bound,single
    ns_sample,unif vs. rwalk
    ns_tol,0.01
    
    ndim = 8

    allesfit_Leonardo_unif/: 
	    75 minutes, 27k samples, logZ = 545.42 +- 0.14
    allesfit_Leonardo_rwalk/: 
	    31 minutes, 24k samples, logZ = 545.36 +- 0.14

#### b)
Fitting all TMNT data together using Dynamic Nested Sampling, once with uniform sampling, once with random-walk sampling:

    ns_modus,dynamic
    ns_nlive,500
    ns_bound,single
    ns_sample,(compared below)
    ns_tol,0.01
    
    ndim = 8

    allesfit_all_TMNT_unif/: 
	    > 24h, aborted 
    allesfit_all_TMNT_rwalk/: 
	    1.3h, 35111 samples, logZ = 1234.26 +- 0.23
    allesfit_all_TMNT_slice/: 
	    > 24h, aborted 
    allesfit_all_TMNT_rslice/: 
	    1h, 28074 samples, logZ = 1233.66 +- 0.23
    allesfit_all_TMNT_hslice/: 
	    16.4h, 30925 samples, logZ = 1231.49 +- 0.23

 - `unif` does not converge within a reasonable run time for higher dimensions. It seems not applicable to our exoplanet scenarios.
 - `rwalk` runs fast, and finds the true solution. It shows quite "conservative" (large) posterior errorbars. This seems to be the best choice for exoplanet modelling.
 - `slice` is theoretically the most robust, but does not converge within a reasonable run time for higher dimensions.
- `rslice` runs fast, but gives somewhat funky posteriors / traceplots. It seems to be overly confident while missing the true solution.
- `hslice` is very slow and gives somewhat funky posteriors / traceplots.

## References

 - Maxted, P. F. L. (2016), Astronomy and Astrophysics, 591, A111
 - Foreman-Mackey, D., Hogg, D. W., Lang, D. & Goodman, J. (2013), Publications of the Astronomical Society of the Pacific, 125, 306
 - Foreman-Mackey, D., Agol, E., Ambikasaran, S. & Angus, R. (2017), The Astronomical Journal, 154, 220
 - Hunter J. D., 2007, Comput. Sci. Eng., 9, 90
 - Jones E. et al., 2001, SciPy: Open Source Scientific tools for Python. Available at: http://www.scipy.org/
 - Rossum G., 1995, Technical Report, Python Reference Manual, Amsterdam, The Netherlands
 - van der Walt S., Colbert S. C., Varoquaux G., 2011, Comput. Sci. Eng., 13, 22



