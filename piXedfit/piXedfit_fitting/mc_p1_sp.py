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


def initfit(gal_z,DL_Gpc,zz,nrands_z):
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
	#if free_z == 1:
	redsh_mod_wave = (1.0+gal_z)*wave
	idx_mod_wave = np.where((redsh_mod_wave>min_spec_wave-30) & (redsh_mod_wave<max_spec_wave+30))

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_chi2_photo_temp = np.zeros(numDataPerRank)
	mod_redcd_chi2_spec_temp = np.zeros(numDataPerRank)
	mod_fluxes_temp = np.zeros((nbands,numDataPerRank))
	mod_spec_flux_temp = np.zeros((nwaves_clean,numDataPerRank))

	count = 0
	for ii in recvbuf_idx:
		# get model spectral fluxes
		str_temp = 'mod/spec/f%d' % int(ii)
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
		poly_legendre = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)

		# apply to the model
		conv_mod_spec_flux_clean = poly_legendre(spec_wave_clean)*conv_mod_spec_flux_clean
		
		chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
		chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
		chi2_spec = np.sum(np.square(chi_spec))

		#chi2 = chi2_photo + chi2_spec
		# weighted reduced chi-square
		#nspecs = len(chi_spec)
		#chi2 = ((nspecs*chi2_photo) + (nbands*chi2_spec))/(nspecs+nbands)
		nspecs = len(chi_spec)
		if nspecs == 0:
			nspecs = 1
		chi2 = (chi2_photo/nbands) + (chi2_spec/nspecs)

		mod_chi2_temp[int(count)] = chi2
		mod_chi2_photo_temp[int(count)] = chi2_photo
		mod_redcd_chi2_spec_temp[int(count)] = chi2_spec/nspecs
		mod_fluxes_temp[:,int(count)] = fluxes  # before normalized
		mod_spec_flux_temp[:,int(count)] = conv_mod_spec_flux_clean/norm # before normalized

		# get parameters
		for pp in range(0,nparams):
			str_temp = 'mod/par/%s' % params[pp]
			if params[pp]=='log_mass' or params[pp]=='log_sfr' or params[pp]=='log_dustmass':
				mod_params_temp[pp][int(count)] = f[str_temp][int(ii)] + log10(norm)
			else:
				mod_params_temp[pp][int(count)] = f[str_temp][int(ii)]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank %d --> progress: z %d of %d (%d%%) and model %d of %d (%d%%)' % (rank,zz+1,nrands_z,(zz+1)*100/nrands_z,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()
	#sys.stdout.write('\n')

	mod_params = np.zeros((nparams,nmodels))
	mod_fluxes = np.zeros((nbands,nmodels))
	mod_spec_flux = np.zeros((nwaves_clean,nmodels))
	mod_chi2 = np.zeros(nmodels)
	mod_chi2_photo = np.zeros(nmodels)
	mod_redcd_chi2_spec = np.zeros(nmodels)

	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_chi2_photo_temp, mod_chi2_photo, root=0)
	comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)

	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)
	for bb in range(0,nbands):
		comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)
	for bb in range(0,nwaves_clean):
		comm.Gather(mod_spec_flux_temp[bb], mod_spec_flux[bb], root=0)

	status_add_err = np.zeros(1)
	modif_obs_flux_err = np.zeros(nbands)
	modif_spec_flux_err_clean = np.zeros(nwaves_clean)

	if rank == 0:
		idx0, min_val = min(enumerate(mod_chi2_photo), key=itemgetter(1))
		fluxes = mod_fluxes[:,idx0]
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)

		#print ("reduced chi2 value of the best-fitting model: %lf" % (mod_chi2[idx0]/nbands))
		if mod_chi2_photo[idx0]/nbands > redcd_chi2:  
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
				chi2 = np.sum(np.square(((norm*fluxes)-obs_fluxes)/modif_obs_flux_err))
				if chi2/nbands <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			#print ("After adding %lf fraction to systematic error, reduced chi2 of best-fit model becomes: %lf" % (sys_err_frac,chi2/nbands))
			status_add_err[0] = 1
		elif mod_chi2_photo[idx0]/nbands <= redcd_chi2:
			status_add_err[0] = 0

		#print ("reduced chi2 value of the best-fitting model: %lf" % (mod_chi2[idx0]/nbands))
		if mod_redcd_chi2_spec[idx0] > redcd_chi2:  
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_spec_flux_err_clean = np.sqrt(np.square(spec_flux_err_clean) + np.square(sys_err_frac*spec_flux_clean))
				chi_spec0 = ((norm*mod_spec_flux[:,idx0])-spec_flux_clean)/modif_spec_flux_err_clean
				chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
				chi2_spec = np.sum(np.square(chi_spec))
				if chi2_spec/len(chi_spec) <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			#print ("After adding %lf fraction to systematic error, reduced chi2 of best-fit model becomes: %lf" % (sys_err_frac,chi2/nbands))
			status_add_err[0] = 1
		elif mod_redcd_chi2_spec[idx0] <= redcd_chi2:
			status_add_err[0] = 0

	comm.Bcast(status_add_err, root=0)

	if status_add_err[0] == 0:
		# Broadcast
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_chi2, root=0)
		 
	elif status_add_err[0] == 1:
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(mod_spec_flux, root=0)
		comm.Bcast(modif_obs_flux_err, root=0) 
		comm.Bcast(modif_spec_flux_err_clean, root=0)

		idx_mpi = np.linspace(0,nmodels-1,nmodels)
		recvbuf_idx = np.empty(numDataPerRank, dtype='d')

		comm.Scatter(idx_mpi, recvbuf_idx, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		mod_params_temp = np.zeros((nparams,numDataPerRank))

		count = 0
		for ii in recvbuf_idx:
			
			if modif_obs_flux_err[0]>0:
				norm = model_leastnorm(obs_fluxes,modif_obs_flux_err,mod_fluxes[:,int(ii)])
			else:
				norm = model_leastnorm(obs_fluxes,obs_flux_err,mod_fluxes[:,int(ii)])
			norm_fluxes = norm*mod_fluxes[:,int(ii)]
			chi_photo = (norm_fluxes-obs_fluxes)/obs_flux_err
			chi2_photo = np.sum(np.square(chi_photo))

			if modif_spec_flux_err_clean[0]>0:
				chi_spec0 = ((norm*mod_spec_flux[:,int(ii)])-spec_flux_clean)/modif_spec_flux_err_clean
			else:
				chi_spec0 = ((norm*mod_spec_flux[:,int(ii)])-spec_flux_clean)/spec_flux_err_clean
			chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
			chi2_spec = np.sum(np.square(chi_spec))

			#chi2 = chi2_photo + chi2_spec
			chi2 = (chi2_photo/nbands) + (chi2_spec/len(chi_spec))

			mod_chi2_temp[int(count)] = chi2

			# get parameters
			for pp in range(0,nparams):
				str_temp = 'mod/par/%s' % params[pp]
				if params[pp]=='log_mass' or params[pp]=='log_sfr' or params[pp]=='log_dustmass':
					mod_params_temp[pp][int(count)] = f[str_temp][int(ii)] + log10(norm)
				else:
					mod_params_temp[pp][int(count)] = f[str_temp][int(ii)]

			count = count + 1

		mod_chi2 = np.zeros(nmodels)
		mod_params = np.zeros(nparams,nmodels)
				
		# gather the scattered data
		comm.Gather(mod_chi2_temp, mod_chi2, root=0)
		for pp in range(0,nparams):
			comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

		# Broadcast
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_params, root=0)

	f.close()

	return mod_params, mod_chi2


## prior function:
def lnprior(theta):
	idx_sel = np.where((theta>=priors_min) & (theta<=priors_max))
	if len(idx_sel[0])==nparams:
		return 0.0
	return -np.inf
 

## function to calculate posterior probability:
def lnprob(theta):
	global obs_fluxes, obs_flux_err

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
	#if free_z == 1:
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
	poly_legendre = np.polynomial.legendre.Legendre.fit(spec_wave_clean, spec_flux_ratio, poly_order)

	# apply to the model
	conv_mod_spec_flux_clean = poly_legendre(spec_wave_clean)*conv_mod_spec_flux_clean
		
	chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
	chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
	chi2_spec = np.sum(np.square(chi_spec))

	# probability
	ln_prob_photo = -0.5*np.sum(np.log(2*pi*obs_flux_err*obs_flux_err)) - 0.5*np.sum(chi2_photo)

	idx1 = np.where((chi_spec0>=lower) & (chi_spec0<=upper))
	derr = spec_flux_err_clean[idx1[0]]
	ln_prob_spec = -0.5*np.sum(np.log(2*pi*derr*derr)) - 0.5*np.sum(chi2_spec)

	nspecs = len(chi_spec)
	if nspecs == 0:
		nspecs = 1
	ln_likeli = (ln_prob_photo/nbands) + (ln_prob_spec/nspecs)

	#idx1 = np.where((chi_spec0>=lower) & (chi_spec0<=upper))
	#m_merge = conv_mod_spec_flux_clean[idx1[0]].tolist() + norm_fluxes.tolist()
	#d_merge = spec_flux_clean[idx1[0]].tolist() + obs_fluxes.tolist()
	#derr_merge = spec_flux_err_clean[idx1[0]].tolist() + obs_flux_err.tolist()
	#ln_likeli = ln_gauss_prob(d_merge,derr_merge,m_merge)

	# get prior:
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
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
def_params_fsps = ['logzsol', 'log_tau', 'log_age', 'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 
				'log_qpah','log_fagn', 'log_tauagn']
params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
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

# number of model SEDs
global nmodels
nmodels = int(f['mod'].attrs['nmodels']/size)*size

# modeling configurations
global imf, sfh_form, dust_ext_law, duste_switch, add_neb_emission, add_agn, gas_logu, fix_dust_index
imf = f['mod'].attrs['imf_type']
sfh_form = f['mod'].attrs['sfh_form']
dust_ext_law = f['mod'].attrs['dust_ext_law']
duste_switch = f['mod'].attrs['duste_switch']
add_neb_emission = f['mod'].attrs['add_neb_emission']
add_agn = f['mod'].attrs['add_agn']
gas_logu = f['mod'].attrs['gas_logu']
params_temp = []
for pp in range(0,int(f['mod'].attrs['nparams_all'])):
	str_temp = 'par%d' % pp 
	params_temp.append(f['mod'].attrs[str_temp])

if duste_switch==1:
	if 'dust_index' in params_temp:
		fix_dust_index = 0 
	else:
		fix_dust_index = 1 
		def_params_val['dust_index'] = f['mod'].attrs['dust_index']
else:
	fix_dust_index = 1

# get number of parameters:
global params, nparams
params, nparams = get_params(free_z, sfh_form, duste_switch, dust_ext_law, add_agn, fix_dust_index)
if rank == 0:
	print ("parameters: ")
	print (params)
	print ("number of parameters: %d" % nparams)

# priors range for the parameters
global priors_min, priors_max
priors_min = np.zeros(nparams)
priors_max = np.zeros(nparams)
for ii in range(0,nparams-1):   # without log_mass
	str_temp1 = 'pr_%s_min' % params[ii]
	str_temp2 = 'pr_%s_max' % params[ii]
	if params[ii]=='z':
		priors_min[ii] = float(config_data['pr_z_min'])
		priors_max[ii] = float(config_data['pr_z_max'])
	else:
		priors_min[ii] = f['mod'].attrs[str_temp1]
		priors_max[ii] = f['mod'].attrs[str_temp2]

# cut model spectrum to match range given by the IFS spectra
#global idx_mod_wave
#if free_z == 0:
#	redsh_mod_wave = (1.0+gal_z)*f['mod/spec/wave'][:]
#	idx_mod_wave = np.where((redsh_mod_wave>min_spec_wave-30) & (redsh_mod_wave<max_spec_wave+30))
#else:
#	idx_mod_wave = []

f.close()

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

#global gauss_likelihood_form
#gauss_likelihood_form = int(config_data['gauss_likelihood_form'])

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
	if dust_ext_law == 0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
	elif dust_ext_law == 1:
		sp.params["dust_type"] = 2  
		sp.params["dust1"] = 0
elif sfh_form==2 or sfh_form==3 or sfh_form==4:
	sp.params["sfh"] = 3
	if dust_ext_law == 0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
	elif dust_ext_law == 1:
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
	mod_params, mod_chi2 = initfit(gal_z,DL_Gpc,0,1)
	idx0, min_val = min(enumerate(mod_chi2), key=itemgetter(1))
	minchi2_params_initfit = mod_params[:,idx0]

elif free_z == 1:
	global rand_z, nrands_z
	nrands_z = int(config_data['nrands_z'])
	rand_z = np.random.uniform(float(config_data['pr_z_min']), float(config_data['pr_z_max']), nrands_z)

	nmodels_merge = int(nmodels*nrands_z)
	mod_params_merge = np.zeros((nparams,nmodels_merge))
	mod_chi2_merge = np.zeros(nmodels_merge)
	for zz in range(0,nrands_z):
		gal_z = rand_z[zz]
		mod_params, mod_chi2 = initfit(gal_z,DL_Gpc,zz,nrands_z)

		for pp in range(0,nparams):
			mod_params_merge[pp,int(zz*nmodels):int((zz+1)*nmodels)] = mod_params[pp]
		mod_chi2_merge[int(zz*nmodels):int((zz+1)*nmodels)] = mod_chi2[:]

	idx0, min_val = min(enumerate(mod_chi2_merge), key=itemgetter(1))
	minchi2_params_initfit = mod_params_merge[:,idx0]

if rank == 0:
	print ("Best-fit parameters from initial fitting:")
	print (minchi2_params_initfit)

# add priors for normalization:
priors_min[int(nparams)-1] = minchi2_params_initfit[int(nparams)-1] - 1.5
priors_max[int(nparams)-1] = minchi2_params_initfit[int(nparams)-1] + 1.5

init_pos = minchi2_params_initfit
width_initpos = 0.08
# modify if an initial point is stack at 0, and it is the lowest value in the prior range
for pp in range(0,nparams):
	temp = init_pos[pp]
	if temp==0 and priors_min[pp]==0:
		init_pos[pp] = width_initpos*(priors_max[pp] - priors_min[pp])
	if temp<=priors_min[pp]:
		init_pos[pp] = priors_min[pp] + (width_initpos*(priors_max[pp] - priors_min[pp]))
	if temp>=priors_max[pp]:
		init_pos[pp] = priors_max[pp] - (width_initpos*(priors_max[pp] - priors_min[pp]))

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
		s.attrs['nparams'] = nparams
		for pp in range(0,nparams):
			str_temp = 'par%d' % pp 
			s.attrs[str_temp] = params[pp]
			s.create_dataset(params[pp], data=np.array(samples[:,pp]))

		o = f.create_group('sed')
		o.create_dataset('flux', data=np.array(obs_fluxes))
		o.create_dataset('flux_err', data=np.array(obs_flux_err))

		sp = f.create_group('spec')
		sp.create_dataset('wave', data=np.array(spec_wave))
		sp.create_dataset('flux', data=np.array(spec_flux))
		sp.create_dataset('flux_err', data=np.array(spec_flux_err))
