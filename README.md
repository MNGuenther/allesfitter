![](docs/source/_static/images/promo.gif)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MNGuenther/allesfitter/master?labpath=allesfitter%2FGUI.ipynb)

[German-ish for *everything-fitter*]

*allesfitter* (Günther & Daylan, 2019, ascl:1903.003) is a public and user-friendly astronomy software package for modeling photometric and RV data. It can accommodate multiple exoplanets, multi-star systems, star spots, stellar flares, and various noise models. A graphical user interface allows to define all input. Then, *allesfitter* automatically runs a nested sampling or MCMC fit, and produces ascii tables, latex tables, and plots. For all this, *allesfitter* constructs an inference framework that unites the versatile packages *ellc* (light curve and RV models; Maxted 2016), *aflare* (flare model; Davenport et al. 2014), *dynesty* (static and dynamic nested sampling; Speagle 2019), *emcee* (Markov Chain Monte Carlo sampling; Foreman-Mackey et al. 2013) and *celerite* (Gaussian Process models; Foreman-Mackey et al. 2017). 
If you use *allesfitter* or parts of it in your work, please cite and acknowledge all software as detailed below.

**Documentation:**

https://www.allesfitter.com/

**Allesfitter citations**:

Please cite both the paper and the code, like `\citep{allesfitter-paper, allesfitter-code}`, with:

    @ARTICLE{allesfitter-paper,
     author = {{G{\"u}nther}, Maximilian N. and {Daylan}, Tansu},
     title = "{Allesfitter: Flexible Star and Exoplanet Inference from Photometry and Radial Velocity}",
     journal = {\apjs},
     keywords = {Exoplanets, Binary stars, Stellar flares, Bayesian statistics, Astronomy software, Starspots, Astronomy data modeling, 498, 154, 1603, 1900, 1855, 1572, 1859, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
     year = 2021,
     month = may,
     volume = {254},
     number = {1},
     eid = {13},
     pages = {13},
     doi = {10.3847/1538-4365/abe70e},
     archivePrefix = {arXiv},
     eprint = {2003.14371},
     primaryClass = {astro-ph.EP},
     adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJS..254...13G},
     adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    
    @MISC{allesfitter-code,
     author = {{G{\"u}nther}, Maximilian~N. and {Daylan}, Tansu},
     title = "{Allesfitter: Flexible Star and Exoplanet Inference From Photometry and Radial Velocity}",
     keywords = {Software },
     howpublished = {Astrophysics Source Code Library},
     year = 2019,
     month = mar,
     archivePrefix = "ascl",
     eprint = {1903.003},
     adsurl = {http://adsabs.harvard.edu/abs/2019ascl.soft03003G},
     adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

**Additional software acknowledgements**:

    - ellc: Maxted, P. F. L. (2016), Astronomy and Astrophysics, 591, A111
    - aflare: Davenport, J. R. A. et al. (2014), The Astrophysical Journal, 797, 122
    - dynesty: Speagel, J. (2019), arXiv:1904.02180
    - emcee: Foreman-Mackey, D., et al. (2013), Publications of the Astronomical Society of the Pacific, 125, 306
    - celerite: Foreman-Mackey, D., et al. (2017), The Astronomical Journal, 154, 220
    - corner: Foreman-Mackey, D., et al. 
    - python: Rossum G. (1995), Technical Report, Python Reference Manual, Amsterdam, The Netherlands
    - numpy: van der Walt S., et al. (2011), Comput. Sci. Eng., 13, 22
    - scipy: Jones E. et al. (2001), SciPy: Open Source Scientific tools for Python. Available at: http://www.scipy.org/
    - matplotlib: Hunter J. D. (2007), Comput. Sci. Eng., 9, 90
    - tqdm: doi:10.5281/zenodo.1468033
    - seaborn: https://seaborn.pydata.org/index.html

**Contributors**: 

Maximilian N. Günther & Tansu Daylan

**License**:

The software is freely available at https://github.com/MNGuenther/allesfitter under the MIT License. Feedback and contributions are very welcome.
