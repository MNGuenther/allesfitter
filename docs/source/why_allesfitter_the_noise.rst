=======================================
The noise
=======================================

Choose from various baseline and noise fitting options (sampling vs. hybrid, GPs, splines, polynomials, and more). Powered by numpy, scipy, and celerite (Foreman-Mackey et al., 2017). In this section, adapted from Günther & Daylan (in prep.), we describe how *allesfitter* can model red noise in different ways, including Gaussian Processes. 

Models used to fit observations can never be a perfect description of the data, even up to white (uncorrelated) noise. This is because observed data in the Universe is always affected by some physical processes not available in the fitted model. The total effect of these unaccounted processes in the data is usually referred to as red (correlated) noise. The apparent correlation of this noise is a consequence of the underlying processes (absent in the fitting model) that generate the associated features in the data.

*allesfitter* includes various options to model red noise, including constant to polynomial terms and spline fitting. We here introduce the most versatile one: using a GP (Rasmussen 2005, Bishop 2006, Roberts 2013) with the squared exponential kernel to non-parametrically model and marginalize over red noise.

Instead of fitting for the parameters of a chosen model (e.g. a polynomial), GP regression fits for a family of functions to determine which one works best.
This is a so-called non-parametric approach.  
In a Bayesian context, a GP can hence be interpreted as a prior on the functions that describe any unknown data set (see e.g. Murphy 2012). By updating it with measurements of the data $D$, one gains the posterior of the model :math:`M`. The GP postulates a family of jointly Gaussian functions, in which the relation of data points are described by the covariance matrix, expressed by the kernel.
A GP can use different kernels and metrics to evaluate the correlation between data points by evaluating the distance between data points with a chosen metric. Certain kernels are well suited to model smooth, long-term variations; others describe more stochastic short-term variations. In practice, the GP is fitted to the data by optimizing its hyperparameters. 

*allesfitter* adapts the *celerite* package, which provides series expressions of typical GP kernels. This enables a significant gain in computing time. A detailed discussion can be found in (Foreman-Mackey et al. 2017).
By fitting hyperparameters of a GP one can model correlations in the data that cannot be explained by the null orbital model. The posterior of these hyperparameters can then be linked to physical sources, such as stellar variability, weather patterns or systematic instrumental noise.