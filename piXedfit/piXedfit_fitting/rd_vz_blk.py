import numpy as np
from math import log10, pow, sqrt 
import sys, os
import h5py
from operator import itemgetter
from mpi4py import MPI
from astropy.io import fits
from astropy.cosmology import *
from scipy.stats import norm as normal
from scipy.stats import t, gamma

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.posteriors import model_leastnorm, calc_chi2, ln_gauss_prob, ln_student_t_prob
from piXedfit.utils.filtering import interp_filters_curves, filtering_interp_filters, cwave_filters, filtering
from piXedfit.utils.redshifting import cosmo_redshifting
from piXedfit.utils.igm_absorption import igm_att_madau, igm_att_inoue


def bayesian_sedfit_gauss(gal_z,zz):
	f = h5py.File(models_spec, 'r')

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# get the spectra
		wave = f['mod/spec/wave'][:]
		str_temp = 'mod/spec/f%d' % idx_parmod_sel[0][int(ii)]
		extnc_spec = f[str_temp][:]

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)

		# IGM absorption:
		if add_igm_absorption == 1:
			if igm_type == 0:
				trans = igm_att_madau(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans
			elif igm_type == 1:
				trans = igm_att_inoue(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans

		# filtering
		fluxes = filtering_interp_filters(redsh_wave,redsh_spec,interp_filters_waves,interp_filters_trans)
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		norm_fluxes = norm*fluxes

		# calculate chi-square and prob
		chi2 = calc_chi2(obs_fluxes,obs_flux_err,norm_fluxes)
		lnlikeli = ln_gauss_prob(obs_fluxes,obs_flux_err,norm_fluxes)

		mod_chi2_temp[int(count)] = chi2
		mod_prob_temp[int(count)] = lnlikeli

		# get parameters
		for pp in range(0,nparams-1):				# exclude z
			str_temp = 'mod/par/%s' % params[pp]
			if params[pp]=='log_mass' or params[pp]=='log_sfr' or params[pp]=='log_dustmass':
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]] + log10(norm)
			else:
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank %d --> progress: z %d of %d (%d%%) and model %d of %d (%d%%)' % (rank,zz+1,nrands_z,(zz+1)*100/nrands_z,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()

	mod_params = np.zeros((nparams,nmodels))
	mod_chi2 = np.zeros(nmodels)
	mod_prob = np.zeros(nmodels)
				
	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)
	for pp in range(0,nparams-1):						# exclude z
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

	comm.Bcast(mod_prob, root=0)
	comm.Bcast(mod_chi2, root=0)
	comm.Bcast(mod_params, root=0)

	# add redshift
	mod_params[int(nparams)-1] = np.zeros(nmodels) + gal_z

	f.close()

	return mod_params, mod_chi2, mod_prob


def bayesian_sedfit_student_t(gal_z,zz):
	f = h5py.File(models_spec, 'r')

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# get the spectra
		wave = f['mod/spec/wave'][:]
		str_temp = 'mod/spec/f%d' % idx_parmod_sel[0][int(ii)]
		extnc_spec = f[str_temp][:]

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)

		# IGM absorption:
		if add_igm_absorption == 1:
			if igm_type == 0:
				trans = igm_att_madau(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans
			elif igm_type == 1:
				trans = igm_att_inoue(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans

		# filtering
		fluxes = filtering_interp_filters(redsh_wave,redsh_spec,interp_filters_waves,interp_filters_trans)
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		norm_fluxes = norm*fluxes

		# calculate chi-square and prob.
		chi2 = calc_chi2(obs_fluxes,obs_flux_err,norm_fluxes)
		chi = (obs_fluxes-norm_fluxes)/obs_flux_err
		lnlikeli = ln_student_t_prob(dof,chi)

		mod_chi2_temp[int(count)] = chi2
		mod_prob_temp[int(count)] = lnlikeli

		# get parameters
		for pp in range(0,nparams-1):				# exclude z
			str_temp = 'mod/par/%s' % params[pp]
			if params[pp]=='log_mass' or params[pp]=='log_sfr' or params[pp]=='log_dustmass':
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]] + log10(norm)
			else:
				mod_params_temp[pp][int(count)] = f[str_temp][idx_parmod_sel[0][int(ii)]]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank %d --> progress: z %d of %d (%d%%) and model %d of %d (%d%%)' % (rank,zz+1,nrands_z,(zz+1)*100/nrands_z,count,numDataPerRank,count*100/numDataPerRank))
		sys.stdout.flush()

	mod_params = np.zeros((nparams,nmodels))
	mod_chi2 = np.zeros(nmodels)
	mod_prob = np.zeros(nmodels)
				
	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)
	for pp in range(0,nparams-1):						# exclude z
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

	comm.Bcast(mod_prob, root=0)
	comm.Bcast(mod_chi2, root=0)
	comm.Bcast(mod_params, root=0)
	
	# add redshift
	mod_params[int(nparams)-1] = np.zeros(nmodels)+gal_z

	f.close()

	return mod_params, mod_chi2, mod_prob


def store_to_fits(sampler_params,mod_chi2,mod_prob,fits_name_out):

	idx, min_val = min(enumerate(mod_chi2), key=itemgetter(1))
	bfit_chi2 = mod_chi2[idx]

	f = h5py.File(models_spec, 'r')
	wave = f['mod/spec/wave'][:]
	idx1 = idx % nmodels 						# modulo
	str_temp = 'mod/spec/f%d' % idx_parmod_sel[0][idx1]
	extnc_spec = f[str_temp][:]
	f.close()
	# best-fit z:
	gal_z = sampler_params['z'][idx]
	redsh_wave,redsh_spec = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=extnc_spec)
	if add_igm_absorption == 1:
		if igm_type == 0:
			trans = igm_att_madau(redsh_wave,gal_z)
			redsh_spec = redsh_spec*trans
		elif igm_type == 1:
			trans = igm_att_inoue(redsh_wave,gal_z)
			redsh_spec = redsh_spec*trans
	fluxes = filtering(redsh_wave,redsh_spec,filters)
	norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
	mod_fluxes = norm*fluxes
	redsh_spec = norm*redsh_spec

	# get initial prediction for stellar mass
	idx_sel = np.where((np.isnan(mod_prob)==False) & (np.isinf(mod_prob)==False))
	array_lnprob = mod_prob[idx_sel[0]] - max(mod_prob[idx_sel[0]])  # normalize
	array_prob = np.exp(array_lnprob)
	array_prob = array_prob/np.sum(array_prob)						 # normalize
	tot_prob = np.sum(array_prob)
	array_val = sampler_params['log_mass'][idx_sel[0]]
	mean_lmass = np.sum(array_val*array_prob)/tot_prob
	mean_lmass2 = np.sum(np.square(array_val)*array_prob)/tot_prob
	std_lmass = sqrt(abs(mean_lmass2 - (mean_lmass**2)))

	# add more if parameters are joint with mass
	if len(params_prior_jtmass)>0:
		for pp in range(0,len(params_prior_jtmass)):
			loc = np.interp(mean_lmass,params_priors[params_prior_jtmass[pp]]['lmass'],params_priors[params_prior_jtmass[pp]]['pval'])
			scale = params_priors[params_prior_jtmass[pp]]['scale']
			mod_prob += np.log(normal.pdf(sampler_params[params_prior_jtmass[pp]],loc=loc,scale=scale))

	for pp in range(0,nparams):
		if params_priors[params[pp]]['form'] == 'gaussian':
			mod_prob += np.log(normal.pdf(sampler_params[params[pp]],loc=params_priors[params[pp]]['loc'],scale=params_priors[params[pp]]['scale']))
		elif params_priors[params[pp]]['form'] == 'studentt':
			mod_prob += np.log(t.pdf(sampler_params[params[pp]],params_priors[params[pp]]['df'],loc=params_priors[params[pp]]['loc'],scale=params_priors[params[pp]]['scale']))
		elif params_priors[params[pp]]['form'] == 'gamma':
			mod_prob += np.log(gamma.pdf(sampler_params[params[pp]],params_priors[params[pp]]['a'],loc=params_priors[params[pp]]['loc'],scale=params_priors[params[pp]]['scale']))
		elif params_priors[params[pp]]['form'] == 'arbitrary':
			mod_prob += np.log(np.interp(sampler_params[params[pp]],params_priors[params[pp]]['values'],params_priors[params[pp]]['prob']))

	crit_chi2 = np.percentile(mod_chi2[np.logical_not(np.isnan(mod_chi2))], perc_chi2)
	idx_sel = np.where((mod_chi2<=crit_chi2) & (np.isnan(mod_prob)==False) & (np.isinf(mod_prob)==False))

	array_lnprob = mod_prob[idx_sel[0]] - max(mod_prob[idx_sel[0]])  # normalize
	array_prob = np.exp(array_lnprob)
	array_prob = array_prob/np.sum(array_prob)						 # normalize
	tot_prob = np.sum(array_prob)

	params_bfits = np.zeros((nparams,2))
	for pp in range(0,nparams):
		if params[pp] == 'log_mass':
			params_bfits[pp][0] = mean_lmass
			params_bfits[pp][1] = std_lmass
		else:
			array_val = sampler_params[params[pp]][idx_sel[0]]
			mean_val = np.sum(array_val*array_prob)/tot_prob
			mean_val2 = np.sum(np.square(array_val)*array_prob)/tot_prob
			std_val = sqrt(abs(mean_val2 - (mean_val**2)))
			params_bfits[pp][0] = mean_val
			params_bfits[pp][1] = std_val

	#==> sampler ids
	nsamples = len(sampler_params[params[0]])
	sampler_id = np.linspace(1, nsamples, nsamples)

	hdr = fits.Header()
	hdr['imf'] = imf
	hdr['nparams'] = nparams
	hdr['sfh_form'] = sfh_form
	hdr['dust_law'] = dust_law
	hdr['nfilters'] = nbands
	hdr['duste_stat'] = duste_switch
	hdr['add_neb_emission'] = add_neb_emission
	if add_neb_emission == 1:
		hdr['gas_logu'] = gas_logu
	hdr['add_agn'] = add_agn
	hdr['add_igm_absorption'] = add_igm_absorption
	hdr['likelihood_form'] = likelihood_form
	if likelihood_form == 'student_t':
		hdr['dof'] = dof
	if add_igm_absorption == 1:
		hdr['igm_type'] = igm_type
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		hdr[str_temp] = filters[bb]
	hdr['free_z'] = 1
	hdr['cosmo'] = cosmo
	hdr['H0'] = H0
	hdr['Om0'] = Om0
	hdr['nrows'] = nsamples
	hdr['redcd_chi2'] = bfit_chi2/nbands
	hdr['perc_chi2'] = perc_chi2
	for pp in range(0,nparams):
		hdr['param%d' % pp] = params[pp]
	hdr['fitmethod'] = 'rdsps'
	hdr['storesamp'] = 1
	hdr['specphot'] = 0
	primary_hdu = fits.PrimaryHDU(header=hdr)

	#==> samplers
	cols0 = []
	col = fits.Column(name='id', format='K', array=np.array(sampler_id))
	cols0.append(col)
	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array(sampler_params[params[pp]]))
		cols0.append(col)
	col = fits.Column(name='chi2', format='D', array=np.array(mod_chi2))
	cols0.append(col)
	col = fits.Column(name='lnprob', format='D', array=np.array(mod_prob))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu1 = fits.BinTableHDU.from_columns(cols, name='samplers')

	#=> best-fitting parameters derived from RDSPS method
	cols0 = []
	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array([params_bfits[pp][0],params_bfits[pp][1]]))
		cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu2 = fits.BinTableHDU.from_columns(cols, name='fit_params')

	#==> Parameters of model with minimum chi-square
	cols0 = []
	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array([sampler_params[params[pp]][idx]]))
		cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu3 = fits.BinTableHDU.from_columns(cols, name='minchi2_params')
	
	#==> observed photometric SED
	cols0 = []
	col = fits.Column(name='flux', format='D', array=np.array(obs_fluxes))
	cols0.append(col)
	col = fits.Column(name='flux_err', format='D', array=np.array(obs_flux_err))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu4 = fits.BinTableHDU.from_columns(cols, name='obs_photo')

	#==> best-fit model photometry
	photo_cwave = cwave_filters(filters)
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(photo_cwave))
	cols0.append(col)
	col = fits.Column(name='flux', format='D', array=np.array(mod_fluxes))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu5 = fits.BinTableHDU.from_columns(cols, name='bfit_photo')

	#==> best-fit model spectrum to the observed photometric SED
	cols0 = []
	col = fits.Column(name='wave', format='D', array=np.array(redsh_wave))
	cols0.append(col)
	col = fits.Column(name='flux', format='D', array=np.array(redsh_spec))
	cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdu6 = fits.BinTableHDU.from_columns(cols, name='bfit_mod_spec')

	hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6])
	hdul.writeto(fits_name_out, overwrite=True)	


"""
USAGE: mpirun -np [npros] python ./rdsps_pcmod.py (1)name_filters_list (2)name_config (3)name_SED_txt (4)name_out_fits
"""

temp_dir = PIXEDFIT_HOME+'/data/temp/'

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

# filters
global filters, nbands
name_filters = str(sys.argv[1])
filters = np.genfromtxt(temp_dir+name_filters, dtype=str)
nbands = len(filters)

global likelihood_form
likelihood_form = config_data['likelihood']

# degree of freedom in the student's t likelihood function, only relevant if likelihood_form='student_t'
global dof
dof = float(config_data['dof'])

# get input SEDs
inputSED_txt = str(sys.argv[3])
f = h5py.File(temp_dir+inputSED_txt, 'r')
n_obs_sed = int(f['obs_seds'].attrs['nbins_calc'])
bulk_obs_fluxes = np.zeros((n_obs_sed,nbands))
bulk_obs_flux_err = np.zeros((n_obs_sed,nbands))
for ii in range(0,n_obs_sed):
	str_temp = 'obs_seds/flux/b%d_f' % ii 
	bulk_obs_fluxes[ii] = f[str_temp][:]

	str_temp = 'obs_seds/flux_err/b%d_ferr' % ii 
	bulk_obs_flux_err[ii] = f[str_temp][:]
f.close()

# add systematic error accommodating various factors, including modeling uncertainty, assume systematic error of 0.1
sys_err_frac = 0.1
bulk_obs_flux_err = np.sqrt(np.square(bulk_obs_flux_err) + np.square(sys_err_frac*bulk_obs_fluxes))

global perc_chi2
perc_chi2 = float(config_data['perc_chi2'])

# names of output FITS files
name_outs = str(sys.argv[4])
name_out_fits = np.genfromtxt(temp_dir+name_outs, dtype=str)

# cosmology
global cosmo, H0, Om0
cosmo = int(config_data['cosmo'])
H0 = float(config_data['H0'])
Om0 = float(config_data['Om0'])

# HDF5 file containing pre-calculated model SEDs
global models_spec
models_spec = config_data['models_spec']

# data of pre-calculated model SEDs
f = h5py.File(models_spec, 'r')

# get list of parameters
global nparams, params
nparams = int(f['mod'].attrs['nparams_all'])  # include all possible parameters
params = []
for pp in range(0,nparams):
	str_temp = 'par%d' % pp
	attrs = f['mod'].attrs[str_temp]
	if isinstance(attrs, str) == False:
		attrs = attrs.decode()
	params.append(attrs)
# add redshift
params.append('z')
nparams = nparams + 1

if rank==0:
	print ("Number of parameters: %d" % nparams)
	print ("List of parameters: ")
	print (params)

# number of model SEDs
global nmodels_parent
nmodels_parent = int(f['mod'].attrs['nmodels'])

# get the prior ranges of the parameters.
# for RDSPS fitting, a fix parameters can't be declared when the fitting run
# it must be applied when building the model rest-frame spectral templates
global priors_min, priors_max
priors_min = np.zeros(nparams)
priors_max = np.zeros(nparams)
for pp in range(0,nparams):
	str_temp1 = "pr_%s_min" % params[pp]
	if str_temp1 in config_data:
		str_temp2 = "pr_%s_max" % params[pp]
		priors_min[pp] = float(config_data[str_temp1])
		priors_max[pp] = float(config_data[str_temp2])
	else:
		str_temp = 'mod/par/%s' % params[pp]
		priors_min[pp] = min(f[str_temp][:])
		priors_max[pp] = max(f[str_temp][:])

# get model indexes satisfying the preferred ranges
status_idx = np.zeros(nmodels_parent)
for pp in range(0,nparams-1): 								# exclude z						
	str_temp = 'mod/par/%s' % params[pp]
	idx0 = np.where((f[str_temp][:]>=priors_min[pp]) & (f[str_temp][:]<=priors_max[pp]))
	status_idx[idx0[0]] = status_idx[idx0[0]] + 1

global idx_parmod_sel, nmodels
idx_parmod_sel = np.where(status_idx==nparams-1) 			# exclude z	
nmodels = int(len(idx_parmod_sel[0])/size)*size
idx_parmod_sel = idx_parmod_sel[0:int(nmodels)]

if rank == 0:
	print ("Number of parent models in models_spec: %d" % nmodels_parent)
	print ("Number of models to be used for fitting: %d" % nmodels)

# modeling configurations
global imf, sfh_form, dust_law, duste_switch, add_neb_emission, add_agn, gas_logu
imf = f['mod'].attrs['imf_type']
sfh_form = f['mod'].attrs['sfh_form']
dust_law = f['mod'].attrs['dust_law']
duste_switch = f['mod'].attrs['duste_switch']
add_neb_emission = f['mod'].attrs['add_neb_emission']
add_agn = f['mod'].attrs['add_agn']
gas_logu = f['mod'].attrs['gas_logu']
f.close()

# get preferred priors
nparams_in_prior = int(config_data['pr_nparams'])
params_in_prior = []
for pp in range(0,nparams_in_prior):
	params_in_prior.append(config_data['pr_param%d' % pp])

global params_priors, params_prior_jtmass
params_priors = {}
params_prior_jtmass = []
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
			params_prior_jtmass.append(params[pp])
	else:
		params_priors[params[pp]]['form'] = 'uniform'

# igm absorption
global add_igm_absorption,igm_type
add_igm_absorption = int(config_data['add_igm_absorption'])
igm_type = int(config_data['igm_type'])

global interp_filters_waves, interp_filters_trans
interp_filters_waves,interp_filters_trans = interp_filters_curves(filters)

global redcd_chi2
redcd_chi2 = 2.0

# generate random redshifts
global rand_z, nrands_z
nrands_z = int(config_data['nrands_z'])
rand_z = np.random.uniform(float(config_data['pr_z_min']), float(config_data['pr_z_max']), nrands_z)

global obs_fluxes, obs_flux_err

# iteration for each SED
for ii in range(0,n_obs_sed):
	obs_fluxes = bulk_obs_fluxes[ii]
	obs_flux_err = bulk_obs_flux_err[ii]

	# iteration for calculations
	nmodels_merge = int(nmodels*nrands_z)
	mod_params_merge = np.zeros((nparams,nmodels_merge))
	mod_chi2_merge = np.zeros(nmodels_merge)
	mod_prob_merge = np.zeros(nmodels_merge)
	for zz in range(0,nrands_z):
		# redshift
		gal_z = rand_z[zz]
		# running the calculation
		if likelihood_form == 'gauss':
			mod_params, mod_chi2, mod_prob = bayesian_sedfit_gauss(gal_z,zz)
		elif likelihood_form == 'student_t':
			mod_params, mod_chi2, mod_prob = bayesian_sedfit_student_t(gal_z,zz)
		else:
			print ("likelihood_form is not recognized!")
			sys.exit()

		for pp in range(0,nparams):
			mod_params_merge[pp,int(zz*nmodels):int((zz+1)*nmodels)] = mod_params[pp][:]
		mod_chi2_merge[int(zz*nmodels):int((zz+1)*nmodels)] = mod_chi2[:]
		mod_prob_merge[int(zz*nmodels):int((zz+1)*nmodels)] = mod_prob[:]

	# change the format to dictionary
	sampler_params = {}
	for pp in range(0,nparams):
		sampler_params[params[pp]] = mod_params_merge[pp]

	# store to fits file
	if rank == 0:
		if name_out_fits.size == 1:
			fits_name_out = name_out_fits.item()
		elif name_out_fits.size > 1:
			fits_name_out = name_out_fits[ii]
		store_to_fits(sampler_params,mod_chi2_merge,mod_prob_merge,fits_name_out)
		sys.stdout.write('\n')
