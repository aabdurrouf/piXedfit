.. _gen_models_seds:

Generating model SEDs
=====================

piXedfit uses the `FSPS <https://github.com/cconroy20/fsps>`_ for modeling the SED of galaxies. With the Python bindings via `Python FSPS <https://dfm.io/python-fsps/current/>`_, generating model SEDs can be done on-the-fly during the SED fitting process. However, some tasks require a generation of model spectra in a fast pace which turn out to be difficult to achieve. These tasks include the generation of model SEDs that will be used in the spectral matching between the Imaging and IFS data, SED fitting with the random dense sampling of parameter space (RDSPS), and the initial fitting (i.e., burning up) before running the SED fitting with the MCMC method. Please note that the MCMC fitting always uses on-the-fly generation of model SEDs.

For that reason, piXedfit provides an option of generating a set of model spectra at rest-frame. The models are stored in HDF5 file format. The model spectra can be generated using function :func:`piXedfit.piXedfit_model.save_models_rest_spec`. Please see the API reference `here <https://pixedfit.readthedocs.io/en/latest/piXedfit_model.html#piXedfit.piXedfit_model.save_models_rest_spec>`_ for more detailed information about this function. In practice, user only need to generate this set of models once, then these models can be used for various further analyses to multiple galaxies.

Generate random model spectra at rest-frame
-------------------------------------------

To generate random model spectra at rest-frame, we can make a script as shown in the following. You can adjust the model configurations depending on the kind of models you need in your analysis. The ranges of parameters can also be adjusted. Please see the API reference of this function `here <https://pixedfit.readthedocs.io/en/latest/piXedfit_model.html#piXedfit.piXedfit_model.save_models_rest_spec>`_. The values of parameters are drawn randomly but designed to be uniformly distributed within desired ranges that can be set in the inputs.  

	.. code-block:: python

		from piXedfit.piXedfit_model import save_models_rest_spec

		imf_type = 1 			# Chabrier (2003)
		sfh_form = 4 			# double power law SFH form
		dust_law = 0 			# Charlot & Fall (2000) dust attenuation law
		duste_switch = 0 		# turn off dust emission
		add_neb_emission = 1 		# turn on nebular emission
		add_agn = 0 			# turn off AGN dusty torus emission

		nmodels = 100000 		# number of model spectra to be generated
		nproc = 20 			# number of processors to be used in the calculation 

		# define ranges of some parameters
		params_range = {'log_age':[-1.0,1.14], 'dust1':[0.0,3.0], 'dust2':[0.0,3.0]}

		name_out = 'model_rest_spec.hdf5'	# name of the output HDF5 file
		save_models_rest_spec(imf_type=imf_type, sfh_form=sfh_form, dust_ext_law=dust_ext_law, 
					duste_switch=duste_switch, add_neb_emission=add_neb_emission, 
					add_agn=add_agn,nmodels=nmodels,nproc=nproc, name_out=name_out) 

The produced models will be used as input in various tasks, including a matching between the imaging and IFS data (see :func:`piXedfit.piXedfit_spectrophotometric.match_imgifs_spectral` and `this example <https://pixedfit.readthedocs.io/en/latest/image_ifs_match.html#spectral-matching>`_), SED fitting with RDSPS method, and initial fitting in the SED fitting with the MCMC method (see API reference of the :ref:`piXedfit_fitting <api_fitting>` module).

.. note::
	It is important to note that all the configurations in the SED modeling and SED fitting are determined in this step. 

Here we determine what types of models that we want to generate and fit to our observed SEDs. This modeling configuration include the choice of initial mass function (IMF; `imf_type`), the choice of star formation history (SFH; `sfh_form`), the choice of dust attenuation law (`dust_law`), whether to switch on/off the following features (nebular emission `add_neb_emission`, dust emission `duste_switch`, AGN dusty torus emission `add_agn`, intergalactic medium `add_igm_absorption`), and the choice of cosmological parameters. Those features have parameters associated with them. This determines the free parameters that will be involved in the SED fitting process.

.. note::
	For the stellar age parameter (`log_age`) to be sufficiently sampled, it is recommended to set a range for `log_age` with minimum value of -1.0 or -2.0 and maximum value that corresponds to the age of the universe at the redshift of the target galaxy.


Generate random model photometric SEDs at observer-frame
--------------------------------------------------------

piXedfit also provides a function for generating a set of model photometric SEDs that are calculated at a given redshift. The models are stored in a FITS file format. This kind of data is not requested as input in most of subsequent analyses. Therefore, this functionality is just a complement to other features that already available in piXedfit.

To generate random model photometric SEDs at observer-frame, we can make a script as shown in the following example. 

	.. code-block:: python

		from piXedfit.piXedfit_model import save_models_photo

		# set of photometric filters
		filters = ['galex_fuv', 'galex_nuv', 'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 
			'sdss_z', '2mass_j', '2mass_h', '2mass_k', 'wise_w1', 'wise_w2', 
			'wise_w3', 'wise_w4', 'spitzer_irac_36', 'spitzer_irac_45', 'spitzer_irac_58', 
			'spitzer_irac_80', 'spitzer_mips_24',  'herschel_pacs_70', 'herschel_pacs_100',
			'herschel_pacs_160', 'herschel_spire_250', 'herschel_spire_350']

		imf_type = 1 			# Chabrier (2003)
		sfh_form = 4 			# double power law SFH form
		dust_law = 0 			# Charlot & Fall (2000) dust attenuation law
		duste_switch = 1 		# turn on dust emission
		add_neb_emission = 1 		# turn on nebular emission
		add_agn = 1 			# turn on AGN dusty torus emission
		add_igm_absorption = 0  	# turn off absoption effect by the intergalactic medium

		# cosmology parameters
		cosmo = 0 			# Flat LCDM
		H0 = 70.0
		Om0 = 0.3

		nmodels = 100000 		# number of model spectra to be generated
		nproc = 20 			# number of processors to be used in the calculation

		gal_z = 0.01

		name_out_fits = 'model_photo_seds.fits'
		save_models_photo(filters=filters, gal_z=gal_z, imf_type=imf_type, sfh_form=sfh_form, 
				dust_ext_law=dust_ext_law, add_igm_absorption=add_igm_absorption, 
				duste_switch=duste_switch, add_neb_emission=add_neb_emission, 
				add_agn=add_agn, nmodels=nmodels, nproc=nproc, cosmo=cosmo, 
				H0=H0, Om0=Om0, name_out_fits=name_out_fits)



