==============================================================================
Installation
==============================================================================


Installing allesfitter
------------------------------------------------------------------------------

- git clone https://github.com/MNGuenther/allesfitter into your ``PYTHONPATH``.
- git clone https://github.com/MNGuenther/exoworlds into your ``PYTHONPATH``.
- (soon to be pip-installable)
 


Requirements
------------------------------------------------------------------------------

Standard packages (install via conda/pip):

- python (>=2.7 or >=3.5)
- numpy
- scikit-learn 
- matplotlib
- h5py
- seaborn
- tqdm

Special packages (install via pip):

- ellc (>=1.8.0) [#f1]_
- dynesty (>=0.9.2) [#f2]_ 
- emcee (>=3.0rc2) [#f3]_
- celerite (>=0.3.0) [#f4]_
- corner (>=2.0.1) 



.. rubric:: Footnotes

.. [#f1] requires a Fortran compiler; on Mac, you can install one e.g. via ``brew install gcc``
.. [#f2] optional; for Nested Sampling
.. [#f3] optional; for MCMC
.. [#f4] optional; for Gaussian Processes

