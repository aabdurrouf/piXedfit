SED fitting
===========

piXedfit provides independent SED fitting functions that are collected in :mod:`piXedfit_fitting` module. The input SED can be either integrated SEDs of galaxies that are fit individually (i.e., one-by-one) or spatially resolved SEDs of a galaxy that are read from `binned data cube <https://pixedfit.readthedocs.io/en/latest/pixel_binning.html#pixel-binning-on-3d-data-cube>`_. For an input of spectrophotometric SED, the SED fitting module can fit simultaneously the photometric SED and the spectrum, providing a powerful constraint for breaking the age--dust--metallicity degeneracies.         

Free parameters
---------------
Before delving into the SED fitting functions, let's see the list of free parameters in the SED fitting. A complete list of parameters and their meaning are given in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...15A/abstract>`_ (Table 1 therein). Each parameter is given with a unique name (i.e., keyword) which also represents how it is sampled in the SED fitting process. Most of the parameters are sampled in logarithmic scale. The keywords of the parameters (with the same order as they appear in the Table 1 of the above paper) are the following: `log_mass`, `logzsol`, `log_age`, `log_tau`, `log_t0`, `log_alpha`, `log_beta`, `dust1`, `dust2`, `dust_index`, `gas_logu`, `log_umin`, `log_gamma`, `log_qpah`, `log_fagn`, and `log_tauagn`. Among the those parameters, only `gas_logu` (ionization parameter in the nebular emissin modeling) is not free in the current version of piXedfit, instead it is fixed in generating model and SED fitting. Parameters that have keyword started with `log` are sampled in logarithmic scale. Only three parameters, `dust1`, `dust2`, and `dust_index`, are sampled in a linear scale. 

.. note::
	Please note that the number of free parameters involved in the SED fitting process is not determined when we configure the SED fitting functions, instead they are determined when we generate model spectral templates (described below).  

Generating model spectra at rest-frame
--------------------------------------
Before performing SED fitting, we need to generate model spectral templates that are calculated at rest-frame of the galaxy (i.e., independent of redshift). Please see the `description <https://pixedfit.readthedocs.io/en/latest/gen_model_SEDs.html#generate-random-model-spectra-at-a-rest-frame>`_ and `tutorial <https://github.com/aabdurrouf/piXedfit/blob/main/examples/Generating_models.ipynb>`_ for detailed information on how to generate this model spectra. 

.. note::
	It is important to note that all the configurations in the SED modeling and SED fitting are determined in this step. 

Here we determine what types of models that we want to generate and fit to our observed SEDs. This modeling configuration include the choice of initial mass function (IMF; `imf_type`), the choice of star formation history (SFH; `sfh_form`), the choice of dust attenuation law (`dust_law`), whether to switch on/off the following features (nebular emission `add_neb_emission`, dust emission `duste_switch`, AGN dusty torus emission `add_agn`, intergalactic medium `add_igm_absorption`). Those features have parameters associated with them. This determines the free parameters that will be involved in the SED fitting process. Please see the API reference `here <https://pixedfit.readthedocs.io/en/latest/piXedfit_model.html#piXedfit.piXedfit_model.save_models_rest_spec>`_ for more detailed information. 

The model parameters are randomly drawn but uniformly distributed within the chosen ranges. This process will generate a set of model spectra that are stored in HDF5 file format. 

.. note::
	It is not necessary to generate a set of model spectra for each galaxy in our sample. 


Fitting methods and some tips
-----------------------------
Overall, piXedfit adopts Bayesian statistics in the SED fitting process. It provides two options for sampling the posterior probability distribution: Markov Chain Monte Carlo (MCMC) and random dense sampling of parameter space (RDSPS). Please refer to `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...15A/abstract>`_ for more detailed information on these two fitting methods. While MCMC is computationally intensive, it is recommended to use this method whenever possible (e.g., sufficient computational resource is available). 

.. warning::
	While the RDSPS method is quite fast (few seconds per SED), this method is very sensitive to our priors (described below). One need to make sure that the selected ranges of parameters are sufficiently wide, especially the parameter age (`log_age`). The derived stellar mass (`log_mass`) from RDSPS fitting is sensitive to the adopted range of `log_age`.

Some tips:
* Generate multiple sets of model spectra (i.e., multiple HDF5 files) for sample of galaxies that have wide range of redshift. 
* For the stellar age parameter (`log_age`) to be sufficiently sampled, it is recommended to set a range for `log_age` with minimum value of -1.0 or -2.0 and maximum value that corresponds to the age of the universe at the redshift of the target galaxy. For larger sample of galaxies, one can produce multiple sets of model spectra with various maximum ages. 
* Since the RDSPS fitting is very sensitive to our priors, it is recommended to use physically motivated priors. This can be in the forms of joint priors among parameters, such as mass-metallicity and mass-age priors that are cnstructed based on empirical scaling relations.  


Defining priors
---------------
Priors for SED fitting can be defined using the :class:`piXedfit.piXedfit_fitting.priors` class. First, we choose ranges of parameters (no need to define it for all parameters). Then, we can choose forms of the priors. There are several forms available, including uniform, Gaussian function, Gamma function, and Student's t function. There is also a choice for defining arbitrary form if users have their own prior form. It is also possible to define joint prior between stellar mass and a particular parameter. 

Please see the API reference `here <https://pixedfit.readthedocs.io/en/latest/piXedfit_fitting.html#piXedfit.piXedfit_fitting.priors>`_ for more information on defining priors.     


Fitting individual SED
----------------------
If one has a single SED or collection of SEDs and intend to fit the SED individually one-by-one, there are two functions available: :func:`piXedfit.piXedfit_fitting.singleSEDfit` and :func:`piXedfit.piXedfit_fitting.singleSEDfit_specphoto`. The former is designed from input photometric SED, while the latter is for input spectrophotometric SED. Please see the API references `here <https://pixedfit.readthedocs.io/en/latest/piXedfit_fitting.html#piXedfit.piXedfit_fitting.singleSEDfit>`_ and `here <https://pixedfit.readthedocs.io/en/latest/piXedfit_fitting.html#piXedfit.piXedfit_fitting.singleSEDfit_specphoto>`_ for more information about these functions. 

The following is an example of script for fitting a single photometric SED.

	.. code-block:: python

		from piXedfit.piXedfit_fitting import singleSEDfit

		# input SED to be fit
		obs_flux = [1.4638932316652199e-16, 1.4038745561641324e-16, 3.8179726118818497e-16, 9.173751654543811e-16, 
			1.12950610607557e-15, 1.1081589447235416e-15, 1.0348753094599238e-15, 7.921053538416444e-16, 
			5.479813400644683e-16, 2.620524286818637e-16, 3.9683859472674366e-17, 1.1746973239557648e-17]
		obs_flux_err = [1.3245499396855221e-17, 5.1672408967693435e-18, 1.0684115183145074e-17, 2.8758039375671638e-18, 
			2.4362783666430513e-18, 2.43151290865674e-18, 6.020149814200945e-18, 1.4838739345484332e-17, 
			1.1592521846969518e-17, 8.156051820519197e-18, 4.044010707857459e-18, 1.104563357898008e-18]
		filters = ['galex_fuv', 'galex_nuv', 'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z', '2mass_j', 
			'2mass_h', '2mass_k', 'wise_w1', 'wise_w2']

		# redshift
		gal_z = 0.0188977

		# Name of HDF5 file containing model spectra at rest-frame
		models_spec = "s_cb_dpl_cf_nde_na_100k.hdf5"

		# call function for defining priors in SED fitting
		from piXedfit.piXedfit_fitting import priors

		# define ranges of some parameters
		ranges = {'logzsol':[-2.0,0.2], 'dust1':[0.0,3.0], 'dust2':[0.0,3.0], 'log_age':[-1.0,1.14]}
		pr = priors(ranges)
		params_ranges = pr.params_ranges()

		# define prior forms
		prior1 = pr.uniform('logzsol')
		prior2 = pr.uniform('dust1')
		prior3 = pr.uniform('dust2')
		prior4 = pr.uniform('log_age')
		params_priors = [prior1, prior2, prior3, prior4]

		# choice of SED fitting method
		fit_method = 'mcmc'

		nproc = 20           # number of cores to be used in the calculation

		# ouptut name
		name_out_fits = "fitting.fits"
		singleSEDfit(obs_flux, obs_flux_err, filters, models_spec, params_ranges=params_ranges, params_priors=params_priors, 
					fit_method=fit_method, gal_z=gal_z, nwalkers=100, nsteps=600, nproc=proc, initfit_nmodels_mcmc=100000, 
					store_full_samplers=1, name_out_fits=name_out_fits) 


Fitting spatially resolved SEDs
-------------------------------
For fitting spatially resolved SEDs in the binned data cube, one can use :func:`piXedfit.piXedfit_fitting.SEDfit_from_binmap` and :func:`piXedfit.piXedfit_fitting.SEDfit_from_binmap_specphoto` functions. The former is designed for photometric data cube, while the latter is for spectrophotometric data cube. Please see the API references of these functions for more information. 

The following is an example of script for performing SED fitting to spectrophotometric data cube. In this script, `binid_range` determines the spatial bins that are going to be fit. With this, we can separate calculations into multiple terminals (i.e., processes) on a multi cores cluster or super computer. Please note that each script also uses multiple cores in parallel computation.

	.. code-block:: python

		from piXedfit.piXedfit_fitting import SEDfit_from_binmap_specphoto

		fits_binmap = "pixbin_corr_specphoto_fluxmap_ngc309.fits"
		binid_range = [0,20]       # range for the ids of spatial bins to be fit

		# Name of HDF5 file containing model spectra at rest-frame
		models_spec = "s_cb_dpl_cf_nde_na_100k.hdf5"

		# call function for defining priors in SED fitting
		from piXedfit.piXedfit_fitting import priors

		# define ranges of some parameters
		ranges = {'logzsol':[-2.0,0.2], 'dust1':[0.0,3.0], 'dust2':[0.0,3.0], 'log_age':[-1.0,1.14]}
		pr = priors(ranges)
		params_ranges = pr.params_ranges()

		# define prior forms
		prior1 = pr.uniform('logzsol')
		prior2 = pr.uniform('dust1')
		prior3 = pr.uniform('dust2')
		prior4 = pr.uniform('log_age')
		params_priors = [prior1, prior2, prior3, prior4]

		# choice of SED fitting method
		fit_method = 'mcmc'

		# range of wavelength in which the spectrum will be fit
		wavelength_range = [3000,9000]

		# spectral resolution in Angstrom
		spec_sigma = 3.5     # spectral resolution of MaNGA

		nproc = 20           # number of cores to be used in the calculation
		SEDfit_from_binmap_specphoto(fits_binmap, binid_range=binid_range, models_spec=models_spec,
						wavelength_range=wavelength_range,params_ranges=params_ranges, 
						params_priors=params_priors,fit_method=fit_method,spec_sigma=spec_sigma, 
						nwalkers=100,nsteps=800,nproc=nproc,initfit_nmodels_mcmc=100000,
						store_full_samplers=1)







