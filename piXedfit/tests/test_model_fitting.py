import os, sys

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)


# Generate model
from piXedfit.piXedfit_model import save_models_rest_spec
imf_type = 1                    # Chabrier (2003) IMF
sfh_form = 4                    # double power law SFH form
dust_law = 0                    # Charlot & Fall (2000) dust attenuation law
duste_switch = 1                # turn off dust emission
add_neb_emission = 1            # turn on nebular emission
add_agn = 0                     # turn off AGN dusty torus emission

nmodels = 10                # number of model spectra to be generated
nproc = 4                      # number of processors to be used in the calculation

# define ranges of some parameters
params_range = {'log_age':[-1.0,1.14], 'dust1':[0.0,3.0], 'dust2':[0.0,3.0]}

name_out = 's_cb_dpl_cf_nde_na_100k.hdf5'     # name of the output HDF5 file
save_models_rest_spec(imf_type=imf_type, sfh_form=sfh_form, dust_law=dust_law, params_range=params_range,
                        duste_switch=duste_switch, add_neb_emission=add_neb_emission, add_agn=add_agn,
                        nmodels=nmodels,nproc=nproc, name_out=name_out)


# Sed fitting
# Fitting spatial bin at the galactic center

# call the function
from piXedfit.piXedfit_fitting import SEDfit_from_binmap
from piXedfit.piXedfit_fitting import priors

def test_SED_fitting():
    fits_binmap = "data/Sample/pixbin_fluxmap_SDSS2MASS.fits"
    binid_range = [0,17]                      # range of the bins to be fit: central bin

    # Name of HDF5 file containing model spectra at rest-frame for initial fitting
    models_spec = "s_cb_dpl_cf_nde_na_100k.hdf5"

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
    fit_method = 'rdsps'
    nproc = 4            # number of cores to be used in calculation
    assert SEDfit_from_binmap(fits_binmap, binid_range=binid_range, models_spec=models_spec,params_ranges=params_ranges, params_priors=params_priors,fit_method=fit_method, nwalkers=100, nsteps=1000, nproc=nproc,initfit_nmodels_mcmc=100000, cosmo=0, H0=70.0, Om0=0.3,store_full_samplers=1) == 1
