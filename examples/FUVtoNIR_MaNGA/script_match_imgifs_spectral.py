from piXedfit.piXedfit_spectrophotometric import match_imgifs_spectral

# script to be executed separately form jupyter notebook
specphoto_file = "specphoto_fluxmap_12073-12703.fits"
nproc = 10                                              # number of cores to be used for calculation
name_out_fits = "corr_%s" % specphoto_file              # desired name for the output FITS file

match_imgifs_spectral(specphoto_file, nproc=nproc, name_out_fits=name_out_fits)
