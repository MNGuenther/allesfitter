# allesfitter

[German-ish for *everything-fitter*]

[beta version]

*allesfitter* is a convenient wrapper around the packages *ellc* (light curve and RV models), *dynesty* (static and dynamic nested sampling) *emcee* (Markov Chain Monte Carlo sampling) and *celerite* (Gaussian Process models).

*allesfitter* is suited for the analysis of multi-planet systems observed with various photometric and RV instruments. It is highly user-friendly, flexible, robust, and allows a wide choice of sampling algorithms and baseline detrending. The user defines all input parameters and settings in a text file, and *allesfitter* automatically runs the nested sampling or MCMC fit, and produces all output such as tables, latex tables, and plots. 

The code is freely available at https://github.com/MNGuenther/allesfitter under the MIT License. Feedback and contributions are very welcome.

If you use *allesfitter* or parts of it in your work, please cite *Günther et al., in prep.* (link). Please also include the following acknowledgement: "This work makes use of the *allesfitter* package (*Günther et al., in prep.*), which is a convenient wrapper around the packages *ellc* (Maxted 2016), *dynesty* (https://github.com/joshspeagle/dynesty), *emcee* (Foreman-Mackey 2013) and *celerite* (Foreman-Mackey 2017). This work makes further use of the *python* programming language (Rossum 1995) and the open-source *python* packages *numpy* (van der Walt, Colbert & Varoquaux 2011), *scipy* (Jones et al. 2001), *matplotlib* (Hunter 2007), *tqdm* (doi:10.5281/zenodo.1468033) and *seaborn* (https://seaborn.pydata.org/index.html)."


## Table of contents
1. Why *allesfitter*?
3. Crash course
3. Installation
4. Fitting 
	4.1 Setting things up
	4.2 Running the fit


## 1. Why allesfitter?
- easy-to-use and all-in-one
- choose between various MCMC and Nested Sampling algorithms (static vs. dynamic, multinest, slicing, ...). Powered by emcee (Foreman-Mackey, 2013) and dynesty (Speagal, 2018). 
- get all that Bayesian evidence! 
- model exoplanets and eclipsing binaries. Powered by ellc (Maxted, 2016).
- globally model any number of photometric and radial velocity observations from various instruments
- choose between multiple baseline and noise fitting options (sampling vs. hybrid, GPs, splines, polynomials, ...)
- fill out a .csv file, click a button, get a coffee, and let *allesfitter* write your paper (it creates all latex tables and plots).


## 2. Crash course

Imagine your photometric survey called "Leonardo" found a planet and you want to model it. The way *allesfitter* works is that you prepare a folder with all your data files (`Leonardo.csv`), a settings file (`settings.csv`) and a parameters file (`params.csv`). Then you let *allesfitter* run on that directory, and it does the rest.

Open the `examples/crash_course` folder. You will see the file `run.py` and the following three folders: `allesfit_Leonardo`, `allesfit_all`, and `simulate_data`.

### `run.py`
This file just contains the simple 3 lines you need to execute to run any of the examples below (after installation). For example:

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


### `allesfit_all`: 

This is an example of fitting the following four data sets:

  - Leonardo.csv (discovery photometry)
  - Michelangelo.csv (follow-up photometry)
  - Donatello.csv (sparse RV data)
  - Raphael.csv (better RV data)

Feel free to explore the `settings.csv` and `params.csv` files to see how to include this additional data into the fit.


### `simulate_data`: 
This folder is only used to create the simulated data set and can be ignored.


## 2. Installation
For now...

- git clone https://github.com/MNGuenther/allesfitter into your PYTHONPATH.
- git clone https://github.com/MNGuenther/lichtkurven into your PYTHONPATH.
- (soon to be pip-installable)
 
Also...

- pip install emcee>=3.0.0 (if you want to run MCMCs)
- git clone https://github.com/joshspeagle/dynesty.git (>=0.9.2b) into your PYTHONPATH (if you want to run Nested Sampling)
- pip install celerite>=0.3.0 (if you want to run Gaussian Processes)
- pip install corner>=2.0.1


## 3. Tutorial

### 3.1 Settings things up
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


### 3.2 Running the fit
- copy run.py from the "examples" that come with allesfitter
- adjust as you need
(todo)


## References

 - Maxted, P. F. L. (2016), Astronomy and Astrophysics, 591, A111
 - Foreman-Mackey, D., Hogg, D. W., Lang, D. & Goodman, J. (2013), Publications of the Astronomical Society of the Pacific, 125, 306
 - Foreman-Mackey, D., Agol, E., Ambikasaran, S. & Angus, R. (2017), The Astronomical Journal, 154, 220
 - Hunter J. D., 2007, Comput. Sci. Eng., 9, 90
 - Jones E. et al., 2001, SciPy: Open Source Scientific tools for Python. Available at: http://www.scipy.org/
 - Rossum G., 1995, Technical Report, Python Reference Manual, Amsterdam, The Netherlands
 - van der Walt S., Colbert S. C., Varoquaux G., 2011, Comput. Sci. Eng., 13, 22

