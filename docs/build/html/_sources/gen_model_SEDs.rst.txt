.. _gen_models_seds:

Generating model SEDs
=====================

piXedfit uses FSPS for modeling the SED of galaxies. With the Python bindings via Python-FSPS, generating model SEDs can be done on the fly in the SED fitting. However, some tasks require a generation of model spectra in a fast pace that is difficult to achieve via the real time calculations. These tasks include generation of model SEDs for reference in the spectral matching between the Imaging and IFS data, SED fitting with the random dense sampling of parameter space (RDSPS), and initial fitting (i.e., burning up) before running SED fitting with the MCMC method (note that the MCMC fitting uses on-the-fly generation of model SEDs).

For that reason, piXedfit provides an option of generating a set of model spectra (in a rest-frame) prior to the analyses. The models are stored in the HDF5 format. The model spectra can be generated using function :func:`piXedfit.piXedfit_model.save_models_rest_spec`. Please see the API reference for more detailed information about this function. In practice, user only need to generate this set of models once, then these models can be used for various further analyses to multiple galaxies.

Generate random model spectra at a rest-frame
---------------------------------------------

To generate random model spectra at a rest-frame, you can make a script like the following. You can adjust the modeling parameters depending on the kind of models you need in your analysis. The ranges of parameters can also be adjusted. Please see the API reference of this function :ref:`here <api_model>`.

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

		name_out = 'model_rest_spec.hdf5'	# name of the output HDF5 file
		save_models_rest_spec(imf_type=imf_type, sfh_form=sfh_form, dust_ext_law=dust_ext_law, 
					duste_switch=duste_switch, add_neb_emission=add_neb_emission, 
					add_agn=add_agn,nmodels=nmodels,nproc=nproc, name_out=name_out) 

The produced models will be used as input in various tasks, including spectral matching of imaging+IFS data (see :func:`piXedfit.piXedfit_spectrophotometric.match_imgifs_spectral`), SED fitting with RDSPS method, and initial fitting for the MCMC method (see API reference of the :ref:`piXedfit_fitting <api_fitting>` module).


Generate random model photometric SEDs at an observer-frame
------------------------------------------------------------

**piXedfit** also provide a functionality of producing a set of model photometric SEDs (calculated at a desired redshift) for a randomly drawn parameters (but uniformly distribution within desired ranges). The models are stored in a FITS file format. This kind of data is not requested as input in most of subsequent analyses. Therefore, this functionality is only a complement to other features provided by **piXedfit**.

To generate random model photometric SEDs at an observer-frame, you can make a script as shown in the following example. 

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



