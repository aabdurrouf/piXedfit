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
from scipy.interpolate import interp1d
from scipy.stats import sigmaclip
from scipy.stats import norm as normal
from scipy.stats import t, gamma

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.posteriors import model_leastnorm, ln_gauss_prob
from piXedfit.utils.redshifting import cosmo_redshifting
from piXedfit.utils.igm_absorption import igm_att_madau, igm_att_inoue
from piXedfit.utils.igm_absorption import igm_att_madau, igm_att_inoue
from piXedfit.piXedfit_model import get_params, get_no_nebem_wave_fit, generate_modelSED_spec_restframe_fit
from piXedfit.piXedfit_model import list_params_fsps, set_initial_fsps, get_params_fsps
from piXedfit.piXedfit_fitting import read_config_file_fit
from piXedfit.piXedfit_spectrophotometric import spec_smoothing


def initfit_fz(gal_z,DL_Gpc):
	# get wavelength free of emission lines
	spec_wave_clean,waveid_excld = get_no_nebem_wave_fit(gal_z,spec_wave,del_wave_nebem)
	spec_flux_clean = np.delete(spec_flux, waveid_excld)
	spec_flux_err_clean = np.delete(spec_flux_err, waveid_excld)

	# open file containing models
	f = h5py.File(models_spec, 'r')

	# get model spectral wavelength 
	wave = f['mod/spec/wave'][:]

	# cut model spectrum to match range given by observed spectrum
	redsh_mod_wave = (1.0+gal_z)*wave
	idx_mod_wave = np.where((redsh_mod_wave>min_spec_wave-100) & (redsh_mod_wave<max_spec_wave+100))

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels_parent_sel-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
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
		
		# cut the model spectrum
		conv_mod_spec_wave,conv_mod_spec_flux = redsh_wave[idx_mod_wave[0]],redsh_spec[idx_mod_wave[0]]

		# get model continuum
		func = interp1d(conv_mod_spec_wave, conv_mod_spec_flux, fill_value='extrapolate')
		conv_mod_spec_flux_clean = func(spec_wave_clean)

		# estimate normalization
		smooth_spec_wave, smooth_mod_spec_flux = spec_smoothing(spec_wave_clean,conv_mod_spec_flux_clean,med_del_wave_sigma)
		smooth_spec_wave, smooth_obs_spec_flux = spec_smoothing(spec_wave_clean,spec_flux_clean,med_del_wave_sigma)
		smooth_spec_wave, smooth_obs_spec_flux_err = spec_smoothing(spec_wave_clean,spec_flux_err_clean,med_del_wave_sigma)
		norm = model_leastnorm(smooth_obs_spec_flux,smooth_obs_spec_flux_err,smooth_mod_spec_flux)
		conv_mod_spec_flux_clean = conv_mod_spec_flux_clean*norm
		
		chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
		chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
		chi2_spec = np.sum(np.square(chi_spec))

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
	mod_redcd_chi2_spec = np.zeros(nmodels)

	# gather the scattered data and collect to rank=0
	comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

	# Broadcast
	comm.Bcast(mod_redcd_chi2_spec, root=0)
	comm.Bcast(mod_params, root=0)

	f.close()

	return mod_params, mod_redcd_chi2_spec


def initfit_vz(gal_z,DL_Gpc,zz,nrands_z):
	# get wavelength free of emission lines
	spec_wave_clean,waveid_excld = get_no_nebem_wave_fit(gal_z,spec_wave,del_wave_nebem)
	spec_flux_clean = np.delete(spec_flux, waveid_excld)
	spec_flux_err_clean = np.delete(spec_flux_err, waveid_excld)
	#nwaves_clean = len(spec_wave_clean)

	# open file containing models
	f = h5py.File(models_spec, 'r')

	# get model spectral wavelength 
	wave = f['mod/spec/wave'][:]

	# cut model spectrum to match range given by observed spectrum
	redsh_mod_wave = (1.0+gal_z)*wave
	idx_mod_wave = np.where((redsh_mod_wave>min_spec_wave-100) & (redsh_mod_wave<max_spec_wave+100))

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels_parent_sel-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
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

		# cut the model spectrum
		conv_mod_spec_wave,conv_mod_spec_flux = redsh_wave[idx_mod_wave[0]],redsh_spec[idx_mod_wave[0]]

		# get model continuum
		func = interp1d(conv_mod_spec_wave, conv_mod_spec_flux, fill_value='extrapolate')
		conv_mod_spec_flux_clean = func(spec_wave_clean)

		# estimate normalization
		smooth_spec_wave, smooth_mod_spec_flux = spec_smoothing(spec_wave_clean,conv_mod_spec_flux_clean,med_del_wave_sigma)
		smooth_spec_wave, smooth_obs_spec_flux = spec_smoothing(spec_wave_clean,spec_flux_clean,med_del_wave_sigma)
		smooth_spec_wave, smooth_obs_spec_flux_err = spec_smoothing(spec_wave_clean,spec_flux_err_clean,med_del_wave_sigma)
		norm = model_leastnorm(smooth_obs_spec_flux,smooth_obs_spec_flux_err,smooth_mod_spec_flux)
		conv_mod_spec_flux_clean = conv_mod_spec_flux_clean*norm
		
		chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
		chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)
		chi2_spec = np.sum(np.square(chi_spec))

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
	mod_redcd_chi2_spec = np.zeros(nmodels)

	# gather the scattered data and collect to rank=0
	comm.Gather(mod_redcd_chi2_spec_temp, mod_redcd_chi2_spec, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

	# Broadcast
	comm.Bcast(mod_redcd_chi2_spec, root=0)
	comm.Bcast(mod_params, root=0)

	f.close()

	return mod_params, mod_redcd_chi2_spec

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
				if params_priors[params[pp]]['mass_sd'] == 1:
					bin_area = float(sys.argv[6])
					loc = np.interp(np.log10(np.power(10.0,theta[nparams-1])/bin_area),params_priors[params[pp]]['lmass'],params_priors[params[pp]]['pval'])
				else:
					loc = np.interp(theta[nparams-1],params_priors[params[pp]]['lmass'],params_priors[params[pp]]['pval'])
				scale = params_priors[params[pp]]['scale']
				lnprior += np.log(normal.pdf(theta[pp],loc=loc,scale=scale))
			
		return lnprior
	return -np.inf

def lnprob(theta):

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
		idx_mod_wave = np.where((redsh_wave>min_spec_wave-100) & (redsh_wave<max_spec_wave+100))

		# IGM absorption:
		if add_igm_absorption == 1:
			if igm_type==0:
				trans = igm_att_madau(redsh_wave,params_val['z'])
				redsh_spec = redsh_spec*trans
			elif igm_type==1:
				trans = igm_att_inoue(redsh_wave,params_val['z'])
				redsh_spec = redsh_spec*trans
			
		# cut and normalize model spectrum
		conv_mod_spec_wave, conv_mod_spec_flux = redsh_wave[idx_mod_wave[0]], redsh_spec[idx_mod_wave[0]]

		# get model continuum
		func = interp1d(conv_mod_spec_wave, conv_mod_spec_flux, fill_value='extrapolate')
		conv_mod_spec_flux_clean = func(spec_wave_clean)
			
		chi_spec0 = (conv_mod_spec_flux_clean-spec_flux_clean)/spec_flux_err_clean
		chi_spec,lower,upper = sigmaclip(chi_spec0, low=spec_chi_sigma_clip, high=spec_chi_sigma_clip)

		ln_likeli = -0.5*np.sum(np.square(chi_spec))

		return lp + ln_likeli


"""
USAGE: mpirun -np [npros] python ./mcmc_pcmod_p1.py (1)conf (2)inputSED_txt (3)data samplers hdf5 file (4)HDF5 file of model spectra (6)bin area (only relevant if prior joint-mass is used and it's in mass surface density)
"""

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

temp_dir = PIXEDFIT_HOME+'/data/temp/'

def_params_fsps, params_assoc_fsps, status_log = list_params_fsps()

# input SED
global spec_wave, spec_flux, spec_flux_err
f = h5py.File(temp_dir+str(sys.argv[2]), 'r')
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

# add systematic error accommodating various factors, including modeling uncertainty, assume systematic error of 0.1
sys_err_frac = 0.1
spec_flux_err = np.sqrt(np.square(spec_flux_err) + np.square(sys_err_frac*spec_flux))

global models_spec
models_spec = sys.argv[4]

# data of pre-calculated model SEDs for initial fitting
f = h5py.File(models_spec, 'r')

# modeling configurations
global imf, sfh_form, dust_law, duste_switch, add_neb_emission, add_agn, nwaves_mod
imf = f['mod'].attrs['imf_type']
sfh_form = f['mod'].attrs['sfh_form']
dust_law = f['mod'].attrs['dust_law']
duste_switch = f['mod'].attrs['duste_switch']
add_neb_emission = f['mod'].attrs['add_neb_emission']
add_agn = f['mod'].attrs['add_agn']
# length of model spectral wavelength, to be used later
nwaves_mod = len(f['mod/spec/wave'][:])

# read configuration file
global cosmo, H0, Om0, nmodels, params, nparams, priors_min, priors_max, params_priors, nrands_z, def_params_val
global smooth_velocity, sigma_smooth, smooth_lsf, name_file_lsf, poly_order, del_wave_nebem, spec_chi_sigma_clip
free_z_temp, free_gas_logz_temp = 1, 1
params0, nparams0 = get_params(free_z_temp, sfh_form, duste_switch, dust_law, add_agn, free_gas_logz_temp)
config_file = str(sys.argv[1])
def_params_val, cosmo, H0, Om0, gal_z, free_z, DL_Gpc, params, fix_params, nparams, priors_min, priors_max, nmodels, params_priors, add_igm_absorption, igm_type, nwalkers, nsteps, nsteps_cut, nrands_z, pr_z_min, pr_z_max, free_gas_logz, smooth_velocity, sigma_smooth, smooth_lsf, name_file_lsf, poly_order, del_wave_nebem, spec_chi_sigma_clip = read_config_file_fit(temp_dir,config_file,params0)

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

global redcd_chi2
redcd_chi2 = 3.0

# check whether smoothing with a line spread function or not
if smooth_lsf == True or smooth_lsf == 1:
	data = np.loadtxt(temp_dir+name_file_lsf)
	lsf_wave, lsf_sigma = data[:,0], data[:,1]
else:
	lsf_wave, lsf_sigma = None, None

# get median spectral resolution (in wavelength space) to be used for smoothing model spectra in the estimation of model normalization
global med_del_wave_sigma
if smooth_lsf == True or smooth_lsf == 1:
	med_del_wave_sigma = 10.0
else:
	if smooth_velocity == 1:
		from astropy import constants as const
		med_del_wave_sigma =  5*0.5*(min_spec_wave+max_spec_wave)*sigma_smooth*1e+3/const.c.value
	else:
		med_del_wave_sigma = 5*sigma_smooth

# call FSPS
global sp 
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)
sp = set_initial_fsps(sp,duste_switch,add_neb_emission,add_agn,sfh_form,dust_law,smooth_velocity=smooth_velocity,
						sigma_smooth=sigma_smooth,smooth_lsf=smooth_lsf,lsf_wave=lsf_wave,lsf_sigma=lsf_sigma)

global params_fsps, nparams_fsps
params_fsps, nparams_fsps = get_params_fsps(params)

# initial fitting
if free_z == 0:
	mod_params, mod_redcd_chi2_spec = initfit_fz(gal_z,DL_Gpc)
	idx0, min_val = min(enumerate(mod_redcd_chi2_spec), key=itemgetter(1))
	minchi2_params_initfit = mod_params[:,idx0]

elif free_z == 1:
	global rand_z
	rand_z = np.random.uniform(pr_z_min, pr_z_max, nrands_z)

	nmodels_merge = int(nmodels*nrands_z)
	mod_params_merge = np.zeros((nparams,nmodels_merge))
	mod_redcd_chi2_spec_merge = np.zeros(nmodels_merge)
	for zz in range(0,nrands_z):
		gal_z = rand_z[zz]
		mod_params, mod_redcd_chi2_spec = initfit_vz(gal_z,DL_Gpc,zz,nrands_z)

		for pp in range(0,nparams):
			mod_params_merge[pp,int(zz*nmodels):int((zz+1)*nmodels)] = mod_params[pp]
		mod_redcd_chi2_spec_merge[int(zz*nmodels):int((zz+1)*nmodels)] = mod_redcd_chi2_spec[:]

	idx0, min_val = min(enumerate(mod_redcd_chi2_spec_merge), key=itemgetter(1))
	minchi2_params_initfit = mod_params_merge[:,idx0]

if rank == 0:
	sys.stdout.write('\n')

# add priors for normalization:
priors_min[int(nparams)-1] = minchi2_params_initfit[int(nparams)-1] - 1.0
priors_max[int(nparams)-1] = minchi2_params_initfit[int(nparams)-1] + 1.0

init_pos = minchi2_params_initfit
width_initpos = 0.08

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
	with h5py.File(sys.argv[3], 'w') as f:
		s = f.create_group('samplers')
		# modeling configuration
		s.attrs['imf'] = imf
		s.attrs['sfh_form'] = sfh_form
		s.attrs['dust_law'] = dust_law
		s.attrs['duste_switch'] = duste_switch
		s.attrs['add_neb_emission'] = add_neb_emission
		s.attrs['add_agn'] = add_agn
		s.attrs['nwaves_mod'] = nwaves_mod
		s.attrs['free_gas_logz'] = free_gas_logz
		s.attrs['smooth_velocity'] = smooth_velocity
		s.attrs['sigma_smooth'] = sigma_smooth
		s.attrs['smooth_lsf'] = smooth_lsf
		s.attrs['name_file_lsf'] = name_file_lsf
		s.attrs['gal_z'] = gal_z
		s.attrs['poly_order'] = poly_order
		s.attrs['del_wave_nebem'] = del_wave_nebem
		s.attrs['spec_chi_sigma_clip'] = spec_chi_sigma_clip
		s.attrs['add_igm_absorption'] = add_igm_absorption
		s.attrs['igm_type'] = igm_type
		s.attrs['nwalkers'] = nwalkers
		s.attrs['nsteps'] = nsteps
		s.attrs['nsteps_cut'] = nsteps_cut
		s.attrs['cosmo'] = cosmo
		s.attrs['H0'] = H0 
		s.attrs['Om0'] = Om0

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

		sp = f.create_group('spec')
		sp.create_dataset('wave', data=np.array(spec_wave))
		sp.create_dataset('flux', data=np.array(spec_flux))
		sp.create_dataset('flux_err', data=np.array(spec_flux_err))

