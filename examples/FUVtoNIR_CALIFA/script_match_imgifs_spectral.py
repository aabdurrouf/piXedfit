import os, sys

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.piXedfit_spectrophotometric import match_imgifs_spectral

specphoto_file = "specphoto_fluxmap_ngc309.fits"
name_out_fits = "corr_%s" % specphoto_file

match_imgifs_spectral(specphoto_file, spec_sigma=2.6, nproc=20, name_out_fits=name_out_fits)