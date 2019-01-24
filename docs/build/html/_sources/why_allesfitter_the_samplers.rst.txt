=======================================
The samplers
=======================================

Choose from various MCMC and Nested Sampling algorithms (static/dynamic, multinest, polychord, slicing). Powered by emcee (Foreman-Mackey et al., 2013) and dynesty (Speagle; `GitHub <https://github.com/joshspeagle/dynesty>`_). In the following section, adapted from Günther & Daylan (in prep.), we describe the inference framework *allesfitter* uses to perform parameter estimation and model selection.



Bayesian statistics and inference
---------------------------------------

Using Bayesian statistics, we assume that there is an underlying probability distribution over the parameters of the model :math:`M` \citep[see e.g.][]{McKay2003} and take fair samples from the posterior probability distribution given the observed photometric and RV data. Bayes' theorem states that the posterior probability of a model $M$ with parameters :math:`\theta$` given some observed data :math:`D` is given by:

:math:`P(\theta|M, D) = \frac{P(D|\theta, M) P(\theta| M)}{P(D|M)}`,

Here, :math:`P(D|\theta, M)` is the probability of observing the data :math:`D` under the model $M$ with parameters :math:`\theta` and is known as the likelihood. Furthermore, :math:`P(\theta| M)$` is the prior probability assigned to the parameter :math:`\theta` of model :math:`M`. In this context, the posterior :math:`P(\theta|M, D)` is the degree of belief about the model and its parameters updated based on observed data :math:`D`. Finally, :math:`P(D|M)` is the marginal likelihood, i.e., the Bayesian evidence,

:math:`P(D|M) = \int P(D|\theta, M) P(\theta| M) \mathrm{d} \theta`.

and quantifies the degree of belief (in the Bayesian sense of probability) one should have about a model :math:`M` given the observed data :math:`D`.
Comparing different physical models, as it is often desired in exoplanet-related studies, relies on the estimation of the Bayesian evidence, :math:`P(D|M)`.

In the context of exoplanet transit modeling, the set of parameters :math:`\theta` contains, for example, the orbital period $P$, epoch of first transit :math:`T_0`, planet radius :math:`R_\mathrm{p}`, stellar radius :math:`R_\star` and more.  The observed data :math:`D` are time series of flux and radial velocity and the times at which these measurements are taken. The choice of priors :math:`P(\theta |M)` can be motivated by previous analyses or scaling arguments. For example, the stellar radius might be constrained by stellar models and parallax measurements by *Gaia*. *allesfitter* reflects this as a Gaussian prior on the associated variable, with mean (maximum probability) at the measured value and the standard deviation reflecting the error bars.



MCMC
---------------------------------------

A Markov chain is a memoryless sequence of elements :math:`{\theta_0, \theta_1, ..., \theta_N}, n\in \mathbb{N}` drawn from a distribution :math:`P(\theta_{n+1}|\theta_n)`, where the realization of :math:`\theta_{n+1}` depends only on the current state, :math:`\theta_{n}`.
A Markov chain can be used to draw fair samples from a probability distribution by choosing an appropriate set of proposals (i.e., transition kernels :math:`P(\theta_{n+1}|\theta_n)`) and ensuring that the stationary distribution of the chain is the desired target probability distribution. 

When sampling from probability distribution functions with MCMC, initial samples are usually not drawn from the posterior due to suboptimal initialization and successive samples can be correlated. Therefore, the initial :math:`X` samples are discarded and the chain is thinned down by a factor of :math:`Y` to ensure that the integrated autocorrelation times of the variables are below 50.

We adopt the *emcee* package, which uses affine invariant sampling (Goodman 2010). This enables efficient sampling from potentially skewed posterior probability distributions with correlated parameters. To do so, it constructs chains from the states of multiple walkers and uses leap-frog proposals to explore the parameter space.
A detailed description of *emcee* can be found in (Foreman-Mackey et al. 2013).

Samples drawn with MCMC are optimized to represent the posterior and hence, for parameter estimation. 
However, estimating the Bayesian evidence with MCMC faces significant challenges, mainly because MCMC is an optimal method for taking samples from the posterior. Calculation of the Bayesian evidence via estimators such as the harmonic mean receives significant contributions from low-posterior samples. Therefore, even though MCMC can efficiently generate samples from the posterior, it cannot accurately estimate the Bayesian evidence. This makes MCMC ineffective in providing a robust model test (Weinberg 2010).



Nested sampling
---------------------------------------

A model comparison can be addressed by using a "Nested Sampling" approach (Skilling 2004, Feroz et al. 2009, Handley et al. 2015) instead of MCMC. In fact, Nested Sampling was developed to compute the Bayesian evidence directly. In the exoplanet context, this then enables to robustly compare radial velocity models with different numbers of exoplanets (Hall et al. 2018), circular versus eccentric orbits, or limb darkening laws.


*allesfitter* implements the *dynesty* package (Speagle; `GitHub <https://github.com/joshspeagle/dynesty>`_), which offers the choice between static and dynamic nested sampling, as well as multiple options such as multinest or polynest algorithms. Our default choise is dynamic nested sampling (Higson et al. 2017). This allows the number of live particles to be changed during sampling, making the integration resolution "dynamic".

