
# allesfitter

![](docs/source/_static/images/teaser.gif)

[German-ish for *everything-fitter*]

*allesfitter* (Günther & Daylan, in prep.) is a public and user-friendly astronomy software package for modeling photometric and RV data. It can accommodate multiple exoplanets, multi-star systems, star spots, stellar flares, and various noise models. A graphical user interface allows to define all input. Then, *allesfitter* automatically runs a nested sampling or MCMC fit, and produces ascii tables, latex tables, and plots. 

For all this, *allesfitter* constructs an inference framework that unites the versatile packages *ellc* (light curve and RV models; Maxted 2016), *aflare* (flare model; Davenport et al. 2014), *dynesty* (static and dynamic nested sampling; https://github.com/joshspeagle/dynesty), *emcee* (Markov Chain Monte Carlo sampling; Foreman-Mackey et al. 2013) and *celerite* (Gaussian Process models; Foreman-Mackey et al. 2017). 

**Documentation:**  
https://allesfitter.readthedocs.io/en/latest/

**Contributors:** 
Maximilian N. Günther, Tansu Daylan

**License:** 
The code is freely available at https://github.com/MNGuenther/allesfitter under the MIT License. Feedback and contributions are very welcome.

**Cite:** 
If you use *allesfitter* or parts of it in your work, please cite *Günther \& Daylan, in prep.* and include the following acknowledgement: "This work makes use of the *allesfitter* package (*Günther \& Daylan, in prep.*), which is a convenient wrapper around the packages *ellc* (Maxted 2016), *aflare1.py* (Davenport 2014), *dynesty* (https://github.com/joshspeagle/dynesty), *emcee* (Foreman-Mackey 2013) and *celerite* (Foreman-Mackey 2017). This work makes further use of the *python* programming language (Rossum 1995) and the open-source *python* packages *numpy* (van der Walt, Colbert & Varoquaux 2011), *scipy* (Jones et al. 2001), *matplotlib* (Hunter 2007), *tqdm* (doi:10.5281/zenodo.1468033) and *seaborn* (https://seaborn.pydata.org/index.html)."