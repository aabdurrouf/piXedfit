Generating model SEDs
=====================

(This page is still under construction!)

Generate random model spectra at a rest-frame
---------------------------------------------

To generate random model spectra at a rest-frame, you can make a script like the following. You can adjust the modeling parameters depending on the kind of models you need in your analysis. 

	.. code-block:: python

		from piXedfit.piXedfit_model import save_models_rest_spec

		imf_type = 1 			# Chabrier (2003)
		sfh_form = 4 			# double power law SFH form
		dust_ext_law = 0 		# Charlot & Fall (2000) dust attenuation law
		duste_switch = 0 		# turn off dust emission
		add_neb_emission = 1 		# turn on nebular emission
		add_agn = 0 			# turn off AGN dusty torus emission

		nmodels = 200000 		# number of model spectra to be generated
		nproc = 20 			# number of processors to be used in the calculation 

		name_out = 'model_rest_spec.hdf5'	# name of the output HDF5 file
		save_models_rest_spec(imf_type=imf_type, sfh_form=sfh_form, dust_ext_law=dust_ext_law, 
					duste_switch=duste_switch, add_neb_emission=add_neb_emission, 
					add_agn=add_agn,nmodels=nmodels,nproc=nproc, name_out=name_out) 


Generate random model photometric SEDs at an observer-frame
------------------------------------------------------------

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
		dust_ext_law = 0 		# Charlot & Fall (2000) dust attenuation law
		duste_switch = 1 		# turn on dust emission
		add_neb_emission = 1 		# turn on nebular emission
		add_agn = 1 			# turn on AGN dusty torus emission
		add_igm_absorption = 0  	# turn off absoption effect by the intergalactic medium

		# cosmology parameters
		cosmo = 0 			# Flat LCDM
		H0 = 70.0
		Om0 = 0.3

		nmodels = 200000 		# number of model spectra to be generated
		nproc = 20 			# number of processors to be used in the calculation

		gal_z = 0.01

		name_out_fits = 'model_photo_seds.fits'
		save_models_photo(filters=filters, gal_z=gal_z, imf_type=imf_type, sfh_form=sfh_form, 
				dust_ext_law=dust_ext_law, add_igm_absorption=add_igm_absorption, 
				duste_switch=duste_switch, add_neb_emission=add_neb_emission, 
				add_agn=add_agn, nmodels=nmodels, nproc=nproc, cosmo=cosmo, 
				H0=H0, Om0=Om0, name_out_fits=name_out_fits)



