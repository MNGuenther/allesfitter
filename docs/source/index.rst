=======================================
allesfitter
=======================================

[German-ish for *everything-fitter*]

*allesfitter* (Günther & Daylan, in prep.) is a public and user-friendly astronomy software package for modeling photometric and RV data. It can accommodate multiple exoplanets, multi-star systems, star spots, stellar flares, and various noise models. A graphical user interface allows to define all input. Then, *allesfitter* automatically runs a nested sampling or MCMC fit, and produces ascii tables, latex tables, and plots. 

For all this, *allesfitter* constructs an inference framework that unites the versatile packages *ellc* (light curve and RV models; Maxted 2016), *aflare* (flare model; Davenport et al. 2014), *dynesty* (static and dynamic nested sampling; https://github.com/joshspeagle/dynesty), *emcee* (Markov Chain Monte Carlo sampling; Foreman-Mackey et al. 2013) and *celerite* (Gaussian Process models; Foreman-Mackey et al. 2017). 

*allesfitter* is publicly available at https://github.com/MNGuenther/allesfitter, feedback and contributions are welcome.

.. image:: _static/images/teaser.gif
   :target: _static/images/teaser.gif



Contents
==================
.. toctree::
   :maxdepth: 2

   why_allesfitter
   cite
   installation
   crash_course
   tutorials
   examples
   performance
   references
