import numpy as np
import sys, os
import fsps
import emcee
import h5py
from math import log10, pow, sqrt, pi 
from operator import itemgetter
from mpi4py import MPI
from astropy.io import fits
from schwimmbad import MPIPool
from astropy.cosmology import *
from scipy.interpolate import interp1d
from scipy.stats import sigmaclip
from scipy.stats import norm as normal
from scipy.stats import t, gamma
from scipy.interpolate import interp1d

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.filtering import interp_filters_curves, filtering_interp_filters 
from piXedfit.utils.posteriors import model_leastnorm, ln_gauss_prob
from piXedfit.piXedfit_model import get_no_nebem_wave_fit, generate_modelSED_spec_restframe_fit
from piXedfit.utils.redshifting import cosmo_redshifting
from piXedfit.utils.igm_absorption import igm_att_madau, igm_att_inoue
from piXedfit.piXedfit_spectrophotometric import spec_smoothing
from piXedfit.piXedfit_fitting import get_params


def initfit_fz(gal_z,DL_Gpc):
	# get wavelength free of emission lines
	spec_wave_clean,waveid_excld = get_no_nebem_wave_fit(gal_z,spec_wave,del_wave_nebem)
	spec_flux_clean = np.delete(spec_flux, waveid_excld)
	spec_flux_err_clean = np.delete(spec_flux_err, waveid_excld)
	nwaves_clean = len(spec_wave_clean)

	# open file containing models
	f = h5py.File(models_spec, 'r')

	# get model spectral wavelength 
	wave = f['mod/spec/wave'][:]

	# cut model spectrum to match range given by observed spectrum
	redsh_mod_wave = (1.0+gal_z)*wave
	idx_mod_wave = np.where((redsh_mod_wave>min_spec_wave-30) & (redsh_mod_wave<max_spec_wave+30))

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels_parent_sel-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_chi2_photo_temp = np.zeros(numDataPerRank)
	mod_redcd_chi2_spec_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# get model spectral fluxes
		str_temp = 'mod/spec/f%d' % idx_parmod_sel[0][int(ii)]
		extnc_spec = f[str_temp][:]

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)

		# IGM absorption:
		if add_igm_absorption == 1:
			if igm_type==0:
				trans = igm_att_madau(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans
			elif igm_type==1:
				trans = igm_att_inoue(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans

		# filtering
		fluxes = filtering_interp_filters(redsh_wave,redsh_spec,interp_filters_waves,interp_filters_trans)
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		norm_fluxes = norm*fluxes

		chi_photo = (norm_fluxes-obs_fluxes)/obs_flux_err
		chi2_photo = np.sum(np.square(chi_photo))
		
		# cut and normalize model spectrum
		# smoothing model spectrum to meet resolution of IFS
		conv_mod_spec_wave,conv_mod_spec_flux = spec_smoothing(redsh_wave[idx_mod_wave[0]],redsh_spec[idx_mod_wave[0]]*norm,spec_sigma)

		# get model continuum
		func = interp1d(conv_mod_spec_wave,conv_mod_spec_flux)
		conv_mod_spec_flux_clean = func(spec_wave_clean)

		# get ratio of the obs and mod continuum
		spec_flux_ratio = spec_flux_clean/conv_mod_spec_flux_clean

		# fit with legendre polynomial
		poly_legendre1 = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)
		poly_legendre2 = np.polynomial.legendre.Legendre.fit(spec_wave_clean, poly_legendre1(spec_wave_clean), 3)

		# apply to the model
		conv_mod_spec_flux_clean = poly_legendre1(spec_wave_clean)*conv_mod_spec_flux_clean/poly_legendre2(spec_wave_clean)
		
		chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
		chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
		chi2_spec = np.sum(np.square(chi_spec))

		mod_chi2_photo_temp[int(count)] = chi2_photo
		mod_redcd_chi2_spec_temp[int(count)] = chi2_spec/len(chi_spec)

		# get parameters
		for pp in range(0,nparams):
			str_temp = 'mod/par/%s' % params[pp]
			if params[pp]=='log_mass':
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]] + log10(norm)
			else:
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank %d --> progress: %d of %d (%d%%)' % (rank,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()

	mod_params = np.zeros((nparams,nmodels))
	mod_chi2_photo = np.zeros(nmodels)
	mod_redcd_chi2_spec = np.zeros(nmodels)

	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_photo_temp, mod_chi2_photo, root=0)
	comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

	# Broadcast
	comm.Bcast(mod_chi2_photo, root=0)
	comm.Bcast(mod_redcd_chi2_spec, root=0)
	comm.Bcast(mod_params, root=0)

	f.close()

	return mod_params, mod_chi2_photo, mod_redcd_chi2_spec


def initfit_vz(gal_z,DL_Gpc,zz,nrands_z):
	# get wavelength free of emission lines
	spec_wave_clean,waveid_excld = get_no_nebem_wave_fit(gal_z,spec_wave,del_wave_nebem)
	spec_flux_clean = np.delete(spec_flux, waveid_excld)
	spec_flux_err_clean = np.delete(spec_flux_err, waveid_excld)
	nwaves_clean = len(spec_wave_clean)

	# open file containing models
	f = h5py.File(models_spec, 'r')

	# get model spectral wavelength 
	wave = f['mod/spec/wave'][:]

	# cut model spectrum to match range given by observed spectrum
	redsh_mod_wave = (1.0+gal_z)*wave
	idx_mod_wave = np.where((redsh_mod_wave>min_spec_wave-30) & (redsh_mod_wave<max_spec_wave+30))

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels_parent_sel-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_chi2_photo_temp = np.zeros(numDataPerRank)
	mod_redcd_chi2_spec_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# get model spectral fluxes
		str_temp = 'mod/spec/f%d' % idx_parmod_sel[0][int(ii)]
		extnc_spec = f[str_temp][:]

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)

		# IGM absorption:
		if add_igm_absorption == 1:
			if igm_type==0:
				trans = igm_att_madau(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans
			elif igm_type==1:
				trans = igm_att_inoue(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans

		# filtering
		fluxes = filtering_interp_filters(redsh_wave,redsh_spec,interp_filters_waves,interp_filters_trans)
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		norm_fluxes = norm*fluxes

		chi_photo = (norm_fluxes-obs_fluxes)/obs_flux_err
		chi2_photo = np.sum(np.square(chi_photo))
		
		# cut and normalize model spectrum
		# smoothing model spectrum to meet resolution of IFS
		conv_mod_spec_wave,conv_mod_spec_flux = spec_smoothing(redsh_wave[idx_mod_wave[0]],redsh_spec[idx_mod_wave[0]]*norm,spec_sigma)

		# get model continuum
		func = interp1d(conv_mod_spec_wave,conv_mod_spec_flux)
		conv_mod_spec_flux_clean = func(spec_wave_clean)

		# get ratio of the obs and mod continuum
		spec_flux_ratio = spec_flux_clean/conv_mod_spec_flux_clean

		# fit with legendre polynomial
		poly_legendre1 = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)
		poly_legendre2 = np.polynomial.legendre.Legendre.fit(spec_wave_clean, poly_legendre1(spec_wave_clean), 3)

		# apply to the model
		conv_mod_spec_flux_clean = poly_legendre1(spec_wave_clean)*conv_mod_spec_flux_clean/poly_legendre2(spec_wave_clean)

		# fit with legendre polynomial
		#poly_legendre = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)

		# apply to the model
		#conv_mod_spec_flux_clean = poly_legendre(spec_wave_clean)*conv_mod_spec_flux_clean
		
		chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
		chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
		chi2_spec = np.sum(np.square(chi_spec))

		mod_chi2_photo_temp[int(count)] = chi2_photo
		mod_redcd_chi2_spec_temp[int(count)] = chi2_spec/len(chi_spec)

		# get parameters
		for pp in range(0,nparams):
			str_temp = 'mod/par/%s' % params[pp]
			if params[pp]=='z':
				mod_params_temp[pp][int(count)] = gal_z
			elif params[pp]=='log_mass':
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]] + log10(norm)
			else:
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank %d --> progress: z %d of %d (%d%%) and model %d of %d (%d%%)' % (rank,zz+1,nrands_z,(zz+1)*100/nrands_z,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()

	mod_params = np.zeros((nparams,nmodels))
	mod_chi2_photo = np.zeros(nmodels)
	mod_redcd_chi2_spec = np.zeros(nmodels)

	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_photo_temp, mod_chi2_photo, root=0)
	comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

	# Broadcast
	comm.Bcast(mod_chi2_photo, root=0)
	comm.Bcast(mod_redcd_chi2_spec, root=0)
	comm.Bcast(mod_params, root=0)

	f.close()

	return mod_params, mod_chi2_photo, mod_redcd_chi2_spec

def lnprior(theta):
	idx_sel = np.where((theta>=priors_min) & (theta<=priors_max))
	if len(idx_sel[0])==nparams:
		lnprior = 0.0
		for pp in range(0,nparams):
			if params_priors[params[pp]]['form'] == 'gaussian':
				lnprior += np.log(normal.pdf(theta[pp],loc=params_priors[params[pp]]['loc'],scale=params_priors[params[pp]]['scale']))
			elif params_priors[params[pp]]['form'] == 'studentt':
				lnprior += np.log(t.pdf(theta[pp],params_priors[params[pp]]['df'],loc=params_priors[params[pp]]['loc'],scale=params_priors[params[pp]]['scale']))
			elif params_priors[params[pp]]['form'] == 'gamma':
				lnprior += np.log(gamma.pdf(theta[pp],params_priors[params[pp]]['a'],loc=params_priors[params[pp]]['loc'],scale=params_priors[params[pp]]['scale']))
			elif params_priors[params[pp]]['form'] == 'arbitrary':
				lnprior += np.log(np.interp(theta[pp],params_priors[params[pp]]['values'],params_priors[params[pp]]['prob']))
			elif params_priors[params[pp]]['form'] == 'joint_with_mass':
				loc = np.interp(theta[nparams-1],params_priors[params[pp]]['lmass'],params_priors[params[pp]]['pval'])
				scale = params_priors[params[pp]]['scale']
				lnprior += np.log(normal.pdf(f[str_temp][idx_parmod_sel[0][int(ii)]],loc=loc,scale=scale))
			
		return lnprior
	return -np.inf

## function to calculate posterior probability:
def lnprob(theta):
	# get prior:
	lp = lnprior(theta)

	idx_sel = np.where((theta>=priors_min) & (theta<=priors_max))

	if np.isfinite(lp)==False or len(idx_sel[0])<nparams:
		return -np.inf
	else:
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = theta[pp]

		# get wavelength free of emission lines
		spec_wave_clean,waveid_excld = get_no_nebem_wave_fit(params_val['z'],spec_wave,del_wave_nebem)
		spec_flux_clean = np.delete(spec_flux, waveid_excld)
		spec_flux_err_clean = np.delete(spec_flux_err, waveid_excld)

		# generate CSP rest-frame spectrum
		rest_wave, rest_flux = generate_modelSED_spec_restframe_fit(sp=sp,sfh_form=sfh_form,params_fsps=params_fsps,params_val=params_val)

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=rest_wave,spec=rest_flux)

		# cut model spectrum to match range given by the observed spectrum
		idx_mod_wave = np.where((redsh_wave>min_spec_wave-30) & (redsh_wave<max_spec_wave+30))

		# IGM absorption:
		if add_igm_absorption == 1:
			if igm_type==0:
				trans = igm_att_madau(redsh_wave,params_val['z'])
				redsh_spec = redsh_spec*trans
			elif igm_type==1:
				trans = igm_att_inoue(redsh_wave,params_val['z'])
				redsh_spec = redsh_spec*trans

		# filtering
		fluxes = filtering_interp_filters(redsh_wave,redsh_spec,interp_filters_waves,interp_filters_trans)
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		norm_fluxes = norm*fluxes
			
		# cut and normalize model spectrum
		# smoothing model spectrum to meet resolution of IFS
		conv_mod_spec_wave,conv_mod_spec_flux = spec_smoothing(redsh_wave[idx_mod_wave[0]],redsh_spec[idx_mod_wave[0]]*norm,spec_sigma)

		# get model continuum
		func = interp1d(conv_mod_spec_wave,conv_mod_spec_flux)
		conv_mod_spec_flux_clean = func(spec_wave_clean)

		# get ratio of the obs and mod continuum
		spec_flux_ratio = spec_flux_clean/conv_mod_spec_flux_clean

		# fit with legendre polynomial
		poly_legendre1 = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)
		poly_legendre2 = np.polynomial.legendre.Legendre.fit(spec_wave_clean, poly_legendre1(spec_wave_clean), 3)

		# apply to the model
		conv_mod_spec_flux_clean = poly_legendre1(spec_wave_clean)*conv_mod_spec_flux_clean/poly_legendre2(spec_wave_clean)

		# fit with legendre polynomial
		#poly_legendre = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)

		# apply to the model
		#conv_mod_spec_flux_clean = poly_legendre(spec_wave_clean)*conv_mod_spec_flux_clean
			
		chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
		chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)

		idx1 = np.where((chi_spec0>=lower) & (chi_spec0<=upper))
		m_merge = conv_mod_spec_flux_clean[idx1[0]].tolist() + norm_fluxes.tolist()
		d_merge = spec_flux_clean[idx1[0]].tolist() + obs_fluxes.tolist()
		derr_merge = spec_flux_err_clean[idx1[0]].tolist() + obs_flux_err.tolist()
		ln_likeli = ln_gauss_prob(d_merge,derr_merge,m_merge)

		return lp + ln_likeli


"""
USAGE: mpirun -np [npros] python ./mcmc_pcmod_p1.py (1)filters (2)conf (3)inputSED_txt (4)data samplers hdf5 file
"""

temp_dir = PIXEDFIT_HOME+'/data/temp/'

# default parameter set
global def_params, def_params_val
def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta','log_age','dust_index','dust1','dust2',
				'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

def_params_val = {'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

global def_params_fsps, params_assoc_fsps, status_log
def_params_fsps = ['logzsol', 'log_tau', 'log_age', 'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 'log_qpah','log_fagn', 'log_tauagn']
params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 'dust_index':"dust_index", 'dust1':"dust1", 
					'dust2':"dust2",'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# configuration file
config_file = str(sys.argv[2])
data = np.genfromtxt(temp_dir+config_file, dtype=str)
config_data = {}
for ii in range(0,len(data[:,0])):
	str_temp = data[:,0][ii]
	config_data[str_temp] = data[:,1][ii]

# get filters
global filters, nbands
name_filters = str(sys.argv[1])
filters = np.genfromtxt(temp_dir+name_filters, dtype=str)
nbands = len(filters)

# input SED
global obs_fluxes, obs_flux_err, spec_wave, spec_flux, spec_flux_err
inputSED_file = str(sys.argv[3])
f = h5py.File(temp_dir+inputSED_file, 'r')
obs_fluxes = f['obs_flux'][:]
obs_flux_err = f['obs_flux_err'][:]
spec_wave = f['spec_wave'][:]
spec_flux = f['spec_flux'][:]
spec_flux_err = f['spec_flux_err'][:]
f.close()

# remove bad spectral fluxes
idx0 = np.where((np.isnan(spec_flux)==False) & (np.isnan(spec_flux_err)==False) & (spec_flux>0) & (spec_flux_err>0))
spec_wave = spec_wave[idx0[0]]
spec_flux = spec_flux[idx0[0]]
spec_flux_err = spec_flux_err[idx0[0]]

# wavelength range of the observed spectrum
global min_spec_wave, max_spec_wave
nwaves = len(spec_wave)
min_spec_wave = min(spec_wave)
max_spec_wave = max(spec_wave)

# spectral resolution
global spec_sigma
spec_sigma = float(config_data['spec_sigma'])

# order of the Legendre polynomial
global poly_order
poly_order = int(config_data['poly_order'])

# add systematic error accommodating various factors, including modeling uncertainty, assume systematic error of 0.1
sys_err_frac = 0.1
obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
spec_flux_err = np.sqrt(np.square(spec_flux_err) + np.square(sys_err_frac*spec_flux))

global del_wave_nebem
del_wave_nebem = float(config_data['del_wave_nebem'])

# clipping for bad spectral points in the chi-square calculation
global spec_chi_sigma_clip
spec_chi_sigma_clip = float(config_data['spec_chi_sigma_clip'])

# cosmology
global cosmo, H0, Om0
cosmo = int(config_data['cosmo'])
H0 = float(config_data['H0'])
Om0 = float(config_data['Om0'])

# redshift
gal_z = float(config_data['gal_z'])
if gal_z<=0.0:
	free_z = 1
	DL_Gpc = 0.0
elif gal_z>0.0:
	free_z = 0
	def_params_val['z'] = gal_z
	# luminosity distance
	if cosmo=='flat_LCDM' or cosmo==0:
		cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
		DL_Gpc0 = cosmo1.luminosity_distance(gal_z)      # in unit of Mpc
	elif cosmo=='WMAP5' or cosmo==1:
		DL_Gpc0 = WMAP5.luminosity_distance(gal_z)
	elif cosmo=='WMAP7' or cosmo==2:
		DL_Gpc0 = WMAP7.luminosity_distance(gal_z)
	elif cosmo=='WMAP9' or cosmo==3:
		DL_Gpc0 = WMAP9.luminosity_distance(gal_z)
	elif cosmo=='Planck13' or cosmo==4:
		DL_Gpc0 = Planck13.luminosity_distance(gal_z)
	elif cosmo=='Planck15' or cosmo==5:
		DL_Gpc0 = Planck15.luminosity_distance(gal_z)
	#elif cosmo=='Planck18' or cosmo==6:
	#	DL_Gpc0 = Planck18.luminosity_distance(gl_z)
	DL_Gpc = DL_Gpc0.value/1.0e+3

# HDF5 file containing pre-calculated model SEDs for initial fitting
global models_spec
models_spec = config_data['models_spec']

# data of pre-calculated model SEDs for initial fitting
f = h5py.File(models_spec, 'r')

# modeling configurations
global imf, sfh_form, dust_law, duste_switch, add_neb_emission, add_agn, gas_logu, nwaves_mod
imf = f['mod'].attrs['imf_type']
sfh_form = f['mod'].attrs['sfh_form']
dust_law = f['mod'].attrs['dust_law']
duste_switch = f['mod'].attrs['duste_switch']
add_neb_emission = f['mod'].attrs['add_neb_emission']
add_agn = f['mod'].attrs['add_agn']
gas_logu = f['mod'].attrs['gas_logu']
# length of model spectral wavelength, to be used later
nwaves_mod = len(f['mod/spec/wave'][:])

# get list of free parameters, their prior ranges, and fix parameters
global params, nparams, priors_min, priors_max
params0, nparams0 = get_params(free_z, sfh_form, duste_switch, dust_law, add_agn)

params = []
fix_params = []
priors_min0 = []
priors_max0 = []
for pp in range(0,nparams0-1): 						# without log_mass
	str_temp1 = 'pr_%s_min' % params0[pp]
	str_temp2 = 'pr_%s_max' % params0[pp]
	if float(config_data[str_temp1]) == float(config_data[str_temp2]):
		def_params_val[params0[pp]] = float(config_data[str_temp1])
		fix_params.append(params0[pp])
	elif float(config_data[str_temp1]) < float(config_data[str_temp2]):
		params.append(params0[pp])
		priors_min0.append(float(config_data[str_temp1]))
		priors_max0.append(float(config_data[str_temp2]))
	else:
		print ("invalid (inverted) range for %s!" % params0[pp])

params.append('log_mass')
nparams = len(params)

priors_min = np.zeros(nparams)
priors_max = np.zeros(nparams)
priors_min[0:nparams-1] = np.asarray(priors_min0)
priors_max[0:nparams-1] = np.asarray(priors_max0)

nfix_params = len(fix_params)
if rank == 0:
	print ("Number of free parameters: %d" % nparams)
	print ("Free parameters: ")
	print (params)
	print ("Number of fix parameters: %d" % nfix_params)
	if nfix_params>0:
		print ("Fix parameters: ")
		print (fix_params)

# number of model SEDs
nmodels_parent = f['mod'].attrs['nmodels']

# get number of models to be used for initil fitting
global nmodels
nmodels = int(config_data['initfit_nmodels_mcmc'])

# get model indexes satisfying the preferred ranges
status_idx = np.zeros(nmodels_parent)
for pp in range(0,nparams-2):						# without log_mass and z
	str_temp = 'mod/par/%s' % params[pp]
	idx0 = np.where((f[str_temp][:]>=priors_min[pp]) & (f[str_temp][:]<=priors_max[pp]))
	status_idx[idx0[0]] = status_idx[idx0[0]] + 1

# and with fix parameters, if any
if nfix_params>0:
	for pp in range(0,nfix_params):
		str_temp = 'mod/par/%s' % fix_params[pp]
		if status_log[fix_params[pp]] == 1 or fix_params[pp]=='logzsol':
			idx0 = np.where((f[str_temp][:]>=def_params_val[fix_params[pp]]-0.1) & (f[str_temp][:]<=def_params_val[fix_params[pp]]+0.1))
		else:
			idx0 = np.where((np.log10(f[str_temp][:])>=np.log10(def_params_val[fix_params[pp]])-0.1) & (np.log10(f[str_temp][:])<=np.log10(def_params_val[fix_params[pp]])+0.1))
		
		status_idx[idx0[0]] = status_idx[idx0[0]] + 1

global idx_parmod_sel, nmodels_parent_sel
idx_parmod_sel = np.where(status_idx==nparams-2+nfix_params)	# without log_mass and z
nmodels_parent_sel = len(idx_parmod_sel[0])

if nmodels>nmodels_parent_sel:
	nmodels = nmodels_parent_sel 

# normalize with size
nmodels = int(nmodels/size)*size

if rank == 0:
	print ("Number of parent models in models_spec: %d" % nmodels_parent_sel)
	print ("Number of models for initial fitting: %d" % nmodels)

f.close()

# get preferred priors
nparams_in_prior = int(config_data['pr_nparams'])
params_in_prior = []
for pp in range(0,nparams_in_prior):
	params_in_prior.append(config_data['pr_param%d' % pp])

global params_priors
params_priors = {}
for pp in range(0,nparams):
	params_priors[params[pp]] = {}
	if params[pp] in params_in_prior:
		params_priors[params[pp]]['form'] = config_data['pr_form_%s' % params[pp]]
		if params_priors[params[pp]]['form'] == 'gaussian':
			params_priors[params[pp]]['loc'] = float(config_data['pr_form_%s_gauss_loc' % params[pp]])
			params_priors[params[pp]]['scale'] = float(config_data['pr_form_%s_gauss_scale' % params[pp]])
		elif params_priors[params[pp]]['form'] == 'studentt':
			params_priors[params[pp]]['df'] = float(config_data['pr_form_%s_stdt_df' % params[pp]])
			params_priors[params[pp]]['loc'] = float(config_data['pr_form_%s_stdt_loc' % params[pp]])
			params_priors[params[pp]]['scale'] = float(config_data['pr_form_%s_stdt_scale' % params[pp]])
		elif params_priors[params[pp]]['form'] == 'gamma':
			params_priors[params[pp]]['a'] = float(config_data['pr_form_%s_gamma_a' % params[pp]])
			params_priors[params[pp]]['loc'] = float(config_data['pr_form_%s_gamma_loc' % params[pp]])
			params_priors[params[pp]]['scale'] = float(config_data['pr_form_%s_gamma_scale' % params[pp]])
		elif params_priors[params[pp]]['form'] == 'arbitrary':
			data = np.loadtxt(temp_dir+config_data['pr_form_%s_arbit_name' % params[pp]])
			params_priors[params[pp]]['values'] = data[:,0]
			params_priors[params[pp]]['prob'] = data[:,1]
		elif params_priors[params[pp]]['form'] == 'joint_with_mass':
			data = np.loadtxt(temp_dir+config_data['pr_form_%s_jtmass_name' % params[pp]])
			params_priors[params[pp]]['lmass'] = data[:,0]
			params_priors[params[pp]]['pval'] = data[:,1]
			params_priors[params[pp]]['scale'] = float(config_data['pr_form_%s_jtmass_scale' % params[pp]])
	else:
		params_priors[params[pp]]['form'] = 'uniform'

# igm absorption
global add_igm_absorption, igm_type
add_igm_absorption = int(config_data['add_igm_absorption'])
igm_type = int(config_data['igm_type'])

global interp_filters_waves, interp_filters_trans
interp_filters_waves,interp_filters_trans = interp_filters_curves(filters)

global redcd_chi2
redcd_chi2 = 2.0

# number of walkers, steps, and nsteps_cut
global nwalkers, nsteps, nsteps_cut 
nwalkers = int(config_data['nwalkers'])
nsteps = int(config_data['nsteps'])
nsteps_cut = int(config_data['nsteps_cut'])

# call FSPS
global sp 
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)
# dust emission
if duste_switch==1:
	sp.params["add_dust_emission"] = True
elif duste_switch==0:
	sp.params["add_dust_emission"] = False
# nebular emission
if add_neb_emission == 1:
	sp.params["add_neb_emission"] = True
	sp.params['gas_logu'] = gas_logu
elif add_neb_emission == 0:
	sp.params["add_neb_emission"] = False
# AGN
if add_agn == 0:
	sp.params["fagn"] = 0
elif add_agn == 1:
	sp.params["fagn"] = 1

if sfh_form==0 or sfh_form==1:
	if sfh_form==0:
		sp.params["sfh"] = 1
	elif sfh_form==1:
		sp.params["sfh"] = 4
	sp.params["const"] = 0
	sp.params["sf_start"] = 0
	sp.params["sf_trunc"] = 0
	sp.params["fburst"] = 0
	sp.params["tburst"] = 30.0
	if dust_law == 0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
	elif dust_law == 1:
		sp.params["dust_type"] = 2  
		sp.params["dust1"] = 0
elif sfh_form==2 or sfh_form==3 or sfh_form==4:
	#sp.params["sfh"] = 3
	if dust_law == 0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
	elif dust_law == 1:
		sp.params["dust_type"] = 2  
		sp.params["dust1"] = 0

global params_fsps, nparams_fsps
params_fsps = []
for ii in range(0,len(def_params_fsps)):
	if def_params_fsps[ii] in params:
		params_fsps.append(def_params_fsps[ii])
nparams_fsps = len(params_fsps)

# initial fitting
if free_z == 0:
	mod_params, mod_chi2_photo, mod_redcd_chi2_spec = initfit_fz(gal_z,DL_Gpc)

	sort_idx = np.argsort(mod_chi2_photo)
	idx0, min_val = min(enumerate(mod_redcd_chi2_spec[sort_idx[0:3]]), key=itemgetter(1))
	mod_id = sort_idx[0:3][idx0]
	minchi2_params_initfit = mod_params[:,int(mod_id)]

elif free_z == 1:
	global rand_z, nrands_z
	nrands_z = int(config_data['nrands_z'])
	rand_z = np.random.uniform(float(config_data['pr_z_min']), float(config_data['pr_z_max']), nrands_z)

	nmodels_merge = int(nmodels*nrands_z)
	mod_params_merge = np.zeros((nparams,nmodels_merge))
	mod_chi2_photo_merge = np.zeros(nmodels_merge)
	mod_redcd_chi2_spec_merge = np.zeros(nmodels_merge)
	for zz in range(0,nrands_z):
		gal_z = rand_z[zz]
		mod_params, mod_chi2_photo, mod_redcd_chi2_spec = initfit_vz(gal_z,DL_Gpc,zz,nrands_z)

		for pp in range(0,nparams):
			mod_params_merge[pp,int(zz*nmodels):int((zz+1)*nmodels)] = mod_params[pp]
		mod_chi2_photo_merge[int(zz*nmodels):int((zz+1)*nmodels)] = mod_chi2_photo[:]
		mod_redcd_chi2_spec_merge[int(zz*nmodels):int((zz+1)*nmodels)] = mod_redcd_chi2_spec[:]

	sort_idx = np.argsort(mod_chi2_photo_merge)
	idx0, min_val = min(enumerate(mod_redcd_chi2_spec_merge[sort_idx[0:3]]), key=itemgetter(1))
	mod_id = sort_idx[0:3][idx0]
	minchi2_params_initfit = mod_params_merge[:,int(mod_id)]

if rank == 0:
	sys.stdout.write('\n')

# add priors for normalization:
priors_min[int(nparams)-1] = minchi2_params_initfit[int(nparams)-1] - 1.5
priors_max[int(nparams)-1] = minchi2_params_initfit[int(nparams)-1] + 1.5

init_pos = minchi2_params_initfit
width_initpos = 0.05

pos = []
for ii in range(0,nwalkers):
	pos_temp = np.zeros(nparams)
	for jj in range(0,nparams):
		width = width_initpos*(priors_max[jj] - priors_min[jj])
		pos_temp[jj] = np.random.normal(init_pos[jj],width,1)
	pos.append(pos_temp)

with MPIPool() as pool:
	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, pool=pool)
	sampler.run_mcmc(pos, nsteps, progress=True)
	samples = sampler.get_chain(discard=nsteps_cut, flat=True)

	# store samplers into HDF5 file
	with h5py.File(sys.argv[4], 'w') as f:
		s = f.create_group('samplers')
		# modeling configuration
		s.attrs['imf'] = imf
		s.attrs['sfh_form'] = sfh_form
		s.attrs['dust_law'] = dust_law
		s.attrs['duste_switch'] = duste_switch
		s.attrs['add_neb_emission'] = add_neb_emission
		s.attrs['add_agn'] = add_agn
		s.attrs['gas_logu'] = gas_logu
		s.attrs['nwaves_mod'] = nwaves_mod
		# free parameters and the prior ranges
		s.attrs['nparams'] = nparams
		for pp in range(0,nparams):
			s.attrs['par%d' % pp] = params[pp]
			s.attrs['min_%s' % params[pp]] = priors_min[pp]
			s.attrs['max_%s' % params[pp]] = priors_max[pp]
			s.create_dataset(params[pp], data=np.array(samples[:,pp]))
		# fix parameters, if any
		s.attrs['nfix_params'] = nfix_params
		if nfix_params>0:
			for pp in range(0,nfix_params): 
				s.attrs['fpar%d' % pp] = fix_params[pp] 
				s.attrs['fpar%d_val' % pp] = def_params_val[fix_params[pp]]

		o = f.create_group('sed')
		o.create_dataset('flux', data=np.array(obs_fluxes))
		o.create_dataset('flux_err', data=np.array(obs_flux_err))

		sp = f.create_group('spec')
		sp.create_dataset('wave', data=np.array(spec_wave))
		sp.create_dataset('flux', data=np.array(spec_flux))
		sp.create_dataset('flux_err', data=np.array(spec_flux_err))

