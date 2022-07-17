SED fitting
===========

piXedfit provides independent SED fitting functions that are collected in :mod:`piXedfit_fitting` module. The input SED can be either integrated SEDs of galaxies that are fit individually (i.e., one-by-one) or spatially resolved SEDs of a galaxy that are read from `binned data cube <https://pixedfit.readthedocs.io/en/latest/pixel_binning.html#pixel-binning-on-3d-data-cube>`_. For an input of spectrophotometric SED, the SED fitting module can fit simultaneously the photometric SED and the spectrum, providing a powerful constraint for breaking the age--dust--metallicity degeneracies.         

Free parameters
---------------
Before delving into the SED fitting functions, let's see the list of free parameters in the SED fitting. A complete list of parameters and their meaning are given in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...15A/abstract>`_ (Table 1 therein). Each parameter is given with a unique name (i.e., keyword) which also represents how it is sampled in the SED fitting process. Most of the parameters are sampled in logarithmic scale. The keywords of the parameters (with the same order as they appear in the Table 1 of the above paper) are the following: `log_mass`, `logzsol`, `log_age`, `log_tau`, `log_t0`, `log_alpha`, `log_beta`, `dust1`, `dust2`, `dust_index`, `gas_logu`, `log_umin`, `log_gamma`, `log_qpah`, `log_fagn`, and `log_tauagn`. Among the those parameters, only `gas_logu` (ionization parameter in the nebular emissin modeling) is not free in the current version of piXedfit, instead it is fixed in generating model and SED fitting. Parameters that have keyword started with `log` are sampled in logarithmic scale. Only three parameters, `dust1`, `dust2`, and `dust_index`, are sampled in a linear scale. 

**Please note that the number of free parameters involved in the SED fitting process is not determined when we configure the SED fitting functions, instead they are determined when we generate model spectral templates (described below)**.  

Generating model spectra at rest-frame
--------------------------------------
Before performing SED fitting, we need to generate model spectral templates that are calculated at the rest-frame (i.e., independent of redshift). Please see the `description <https://pixedfit.readthedocs.io/en/latest/gen_model_SEDs.html#generate-random-model-spectra-at-a-rest-frame>`_ and `tutorial <https://github.com/aabdurrouf/piXedfit/blob/main/examples/Generating_models.ipynb>`_ for detailed information on how to generate this model spectra. It is important to note that   


Fitting methods and some tips
-----------------------------
Overall, piXedfit adopts Bayesian statistics in the SED fitting process. It provides two options for sampling the posterior probability distribution: Markov Chain Monte Carlo (MCMC) and random dense sampling of parameter space (RDSPS). Please refer to `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...15A/abstract>`_ for more detailed information on these two fitting methods. It is recmmended to use the MCMC method whenever possible (e.g., sufficient computational resource is available).    


Defining priors
---------------


Fitting individual SED
----------------------




Fitting spatially resolved SEDs
-------------------------------

