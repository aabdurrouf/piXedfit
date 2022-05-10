import numpy as np
import sys, os
import fsps
import emcee
import h5py
from math import log10 
from operator import itemgetter
from mpi4py import MPI
from astropy.io import fits
from schwimmbad import MPIPool
from astropy.cosmology import *

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.filtering import interp_filters_curves, filtering_interp_filters 
from piXedfit.utils.posteriors import calc_modchi2_leastnorm, model_leastnorm, gauss_ln_prob, calc_chi2
from piXedfit.piXedfit_model import generate_modelSED_photo_fit, generate_modelSED_spec_fit
from piXedfit.piXedfit_fitting import get_params
from piXedfit.utils.redshifting import cosmo_redshifting


def initfit(gal_z,DL_Gpc):
	f = h5py.File(models_spec, 'r')

	numDataPerRank = int(nmodels/size)
	idx_mpi = np.linspace(0,nmodels-1,nmodels)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')

	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_fluxes_temp = np.zeros((nbands,numDataPerRank))

	count = 0
	for ii in recvbuf_idx:
		# get the spectra
		wave = f['mod/spec/wave'][:]
		str_temp = 'mod/spec/f%d' % int(ii)
		extnc_spec = f[str_temp][:]

		# redshifting
		redsh_wave,redsh_spec = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,
													z=gal_z,wave=wave,spec=extnc_spec)

		# IGM absorption:
		if add_igm_absorption == 1:
			if igm_type == 0 or igm_type == 'madau1995':
				trans = igm_att_madau(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans
			elif igm_type == 1 or igm_type == 'inoue2014':
				trans = igm_att_inoue(redsh_wave,gal_z)
				redsh_spec = redsh_spec*trans

		# filtering
		fluxes = filtering_interp_filters(redsh_wave,redsh_spec,interp_filters_waves,interp_filters_trans)
		norm = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		norm_fluxes = norm*fluxes

		# calculate chi-square and prob
		chi2 = calc_chi2(obs_fluxes,obs_flux_err,norm_fluxes)

		mod_chi2_temp[int(count)] = chi2
		mod_fluxes_temp[:,int(count)] = fluxes  # before normalized

		# get parameters
		for pp in range(0,nparams):
			str_temp = 'mod/par/%s' % params[pp]
			if params[pp]=='z':
				mod_params_temp[pp][int(count)] = gal_z
			elif params[pp]=='log_mass':
				mod_params_temp[pp][int(count)] = f[str_temp][int(ii)] + log10(norm)
			else:
				mod_params_temp[pp][int(count)] = f[str_temp][int(ii)]

		count = count + 1

		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	sys.stdout.write('\n')

	mod_params = np.zeros((nparams,nmodels))
	mod_fluxes = np.zeros((nbands,nmodels))
	mod_chi2 = np.zeros(nmodels)
				
	# gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)
	for bb in range(0,nbands):
		comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)

	status_add_err = np.zeros(1)
	modif_obs_flux_err = np.zeros(nbands)

	if rank == 0:
		idx0, min_val = min(enumerate(mod_chi2), key=itemgetter(1))

		fluxes = mod_fluxes[:,idx0]

		#print ("reduced chi2 value of the best-fitting model: %lf" % (mod_chi2[idx0]/nbands))
		if mod_chi2[idx0]/nbands > redcd_chi2:  
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
				chi2 = calc_modchi2_leastnorm(obs_fluxes,modif_obs_flux_err,fluxes)
				if chi2/nbands <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			#print ("After adding %lf fraction to systematic error, reduced chi2 of best-fit model becomes: %lf" % (sys_err_frac,chi2/nbands))
			status_add_err[0] = 1
		elif mod_chi2[idx0]/nbands <= redcd_chi2:
			status_add_err[0] = 0

	comm.Bcast(status_add_err, root=0)

	if status_add_err[0] == 0:
		# Broadcast
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(mod_chi2, root=0)
		 
	elif status_add_err[0] == 1:
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(modif_obs_flux_err, root=0) 

		idx_mpi = np.linspace(0,nmodels-1,nmodels)
		recvbuf_idx = np.empty(numDataPerRank, dtype='d')

		comm.Scatter(idx_mpi, recvbuf_idx, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		mod_params_temp = np.zeros((nparams,numDataPerRank))

		count = 0
		for ii in recvbuf_idx:
			fluxes = mod_fluxes[:,int(ii)]

			norm = model_leastnorm(obs_fluxes,modif_obs_flux_err,fluxes)
			norm_fluxes = norm*fluxes

			# calculate chi-square and prob
			chi2 = calc_chi2(obs_fluxes,modif_obs_flux_err,norm_fluxes)

			mod_chi2_temp[int(count)] = chi2

			# get parameters
			for pp in range(0,nparams):
				str_temp = 'mod/par/%s' % params[pp]
				if params[pp]=='log_mass':
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

	# model fluxes
	photo_SED_flux = generate_modelSED_photo_fit(sp=sp,sfh_form=sfh_form,filters=filters,add_igm_absorption=add_igm_absorption,
											igm_type=igm_type,params_fsps=params_fsps,params_val=params_val,
											DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,
											interp_filters_waves=interp_filters_waves,
											interp_filters_trans=interp_filters_trans)

	if gauss_likelihood_form == 0:
		ln_likeli = gauss_ln_prob(obs_fluxes,obs_flux_err,photo_SED_flux)
	elif gauss_likelihood_form == 1:
		chi2 = calc_chi2(obs_fluxes,obs_flux_err,photo_SED_flux)
		ln_likeli = -0.5*chi2

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
global obs_fluxes, obs_flux_err
inputSED_txt = str(sys.argv[3])
data = np.loadtxt(temp_dir+inputSED_txt)
obs_fluxes = np.asarray(data[:,0])
obs_flux_err = np.asarray(data[:,1])

# add systematic error accommodating various factors, including modeling uncertainty, assume systematic error of 0.1
sys_err_frac = 0.1
obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))

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

if duste_switch=='duste' or duste_switch==1:
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

f.close()

# igm absorption
global add_igm_absorption,igm_type
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

global gauss_likelihood_form
gauss_likelihood_form = int(config_data['gauss_likelihood_form'])

# call FSPS
global sp 
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)
# dust emission
if duste_switch == 'duste':
	sp.params["add_dust_emission"] = True
elif duste_switch == 'noduste':
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

if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh':
	if sfh_form == 'tau_sfh':
		sp.params["sfh"] = 1
	elif sfh_form == 'delayed_tau_sfh':
		sp.params["sfh"] = 4
	sp.params["const"] = 0
	sp.params["sf_start"] = 0
	sp.params["sf_trunc"] = 0
	sp.params["fburst"] = 0
	sp.params["tburst"] = 30.0
	if dust_ext_law == 'CF2000' :
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
	elif dust_ext_law == 'Cal2000':
		sp.params["dust_type"] = 2  
		sp.params["dust1"] = 0
elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh':
	sp.params["sfh"] = 3
	if dust_ext_law == 'CF2000' :
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
	elif dust_ext_law == 'Cal2000':
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
	mod_params, mod_chi2 = initfit(gal_z,DL_Gpc)
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
		mod_params, mod_chi2 = initfit(gal_z,DL_Gpc)

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



