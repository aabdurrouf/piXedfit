Spatial and Spectral Matching of imaging+IFS data
=================================================

(This page is still under constraction!)



Spatial matching
----------------

	.. code-block:: python

		from piXedfit.piXedfit_spectrophotometric import match_imgifs_spatial

		photo_fluxmap = "fluxmap_ngc309.fits"			# photometric data cube
		ifs_data = "NGC0309.COMB.rscube.fits.gz"		# IFS data cube
		ifs_survey = "califa"					# IFS data source
		name_out_fits = "specphoto_fluxmap_ngc309.fits"		# name of output file
		match_imgifs_spatial(photo_fluxmap, ifs_data, ifs_survey=ifs_survey, 
					name_out_fits=name_out_fits)



Spectral matching
-----------------
.. _spectral_matching:

	.. code-block:: python

		from piXedfit.piXedfit_spectrophotometric import match_imgifs_spectral

		# spectrophotometric data cube produced in the previous step
		specphoto_file = "specphoto_fluxmap_ngc309.fits"		 
		name_out_fits = "corr_%s" % specphoto_file 		# name of output file

		match_imgifs_spectral(specphoto_file, nproc=20, name_out_fits=name_out_fits)