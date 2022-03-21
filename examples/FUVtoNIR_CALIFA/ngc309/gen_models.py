import sys, os

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.piXedfit_model import save_models


filters = ['galex_fuv', 'galex_nuv', 'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 
           'sdss_z', '2mass_j', '2mass_h', '2mass_k', 'wise_w1', 'wise_w2']


imf_type = 1				# Chabrier (2003)
sfh_form = 4				# Double power-law SFH
dust_ext_law = 1			# Calzetti et al. (2000)

cosmo = 0					# flat_LCDM
H0 = 70.0
Om0 = 0.3

logzsol_range = [-2.0,0.2]
log_tau_range = [-1.5,1.14]
log_age_range = [-1.0,1.14]
log_alpha_range = [-2.0,2.0]
log_beta_range = [-2.0,2.0]
dust2_range = [0.0,3.0]

nproc = 10					# Number of cores/processors
npmod_seds = 100000			# Number of model SEDs to be generated

# Redshift
gal_z = 0.0188977

name_out_fits = "ngc309_models.fits"
save_models(gal_z=gal_z,filters=filters,imf_type=imf_type,sfh_form=sfh_form,dust_ext_law=dust_ext_law,
							npmod_seds=npmod_seds,logzsol_range=logzsol_range,log_tau_range=log_tau_range,
							log_age_range=log_age_range,dust2_range=dust2_range,log_alpha_range=log_alpha_range,
							log_beta_range=log_beta_range,nproc=nproc,cosmo=cosmo,H0=H0,Om0=Om0,name_out_fits=name_out_fits)

