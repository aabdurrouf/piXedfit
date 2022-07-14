from piXedfit.piXedfit_spectrophotometric import match_imgifs_spectral

specphoto_file = "specphoto_fluxmap_ngc309.fits"
nproc = 10
name_out_fits = "corr_%s" % specphoto_file
match_imgifs_spectral(specphoto_file, nproc=nproc, name_out_fits=name_out_fits)
