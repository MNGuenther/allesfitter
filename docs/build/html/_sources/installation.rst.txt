=======================================
Installation
=======================================



Installing allesfitter
---------------------------------------

- git clone https://github.com/MNGuenther/allesfitter into your ``PYTHONPATH``.
- git clone https://github.com/MNGuenther/exoworlds into your ``PYTHONPATH``.
- (soon to be pip-installable)
 


Requirements
---------------------------------------

- python >=2.7 or >=3.5
- numpy (install via conda or pip)
- matplotlib (install via conda or pip)
- seaborn (install via conda or pip)
- tqdm (install via conda or pip)
- ellc (>=1.8.0; install via pip; this requires that a Fortran compiler is installed. If it is missing, you can use e.g. homebrew with ``brew install gcc`` to install one on Mac.)
- dynesty (git clone https://github.com/joshspeagle/dynesty.git (>=0.9.2b) into your ``PYTHONPATH``; then install via ``run python setup.py install``; if you want Nested Sampling)
- emcee (>=3.0.0; install via pip; if you want MCMC)
- celerite (>=0.3.0; install via pip; if you want Gaussian Processes)
- corner (>=2.0.1; install via pip)
