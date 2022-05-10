import numpy as np
import sys, os
import fsps
import emcee
from operator import itemgetter
from mpi4py import MPI
from astropy.io import fits
from schwimmbad import MPIPool
from astropy.cosmology import *

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.filtering import match_filters_array
from piXedfit.utils.posteriors import calc_modchi2_leastnorm, model_leastnorm, gauss_ln_prob, calc_chi2
from piXedfit.piXedfit_model import generate_modelSED_photo_fit, generate_modelSED_spec_fit
from piXedfit.piXedfit_fitting import get_params


# function for initial fitting:
# mod_fluxes0 = [idx-band][idx-model]  --> this input will be used if increase_ferr==0
# mod_params1 = [idx-param][idx-model]  --> this input will be used if increase_ferr==0
def initfit(increase_ferr,name_saved_randmod):
	# Fits file containing pre-calculated model SEDs:
	hdu = fits.open(name_saved_randmod)
	data_randmod = hdu[1].data
	hdu.close()

	# define reduced chi2 to be achieved
	redcd_chi2 = 2.0
	
	numDataPerRank = int(npmod_seds/size)

	if increase_ferr == 1:
		
		idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
		# allocate memory in each process
		recvbuf_idx = np.empty(numDataPerRank, dtype='d')     # allocate space for recvbuf
		# scatter the ids to the processes
		comm.Scatter(idx_mpi, recvbuf_idx, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)

		count = 0
		for ii in recvbuf_idx:
			# get model fluxes
			fluxes = np.zeros(nbands)
			for bb in range(0,nbands):
				fluxes[bb] = data_randmod[filters[bb]][int(ii)]

			chi2 = calc_modchi2_leastnorm(obs_fluxes,obs_flux_err,fluxes)
			
			mod_chi2_temp[int(count)] = chi2

			count = count + 1

			sys.stdout.write('\r')
			sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
			sys.stdout.flush()
		sys.stdout.write('\n')

		mod_chi2 = None
		if rank == 0:
			mod_chi2 = np.empty(numDataPerRank*size, dtype='d')
				
		# gather the scattered data
		comm.Gather(mod_chi2_temp, mod_chi2, root=0)

		if rank == 0:
			# allocate memory:
			modif_obs_flux_err = np.zeros(nbands)   		# default: 0
			params_chi2min_initfit = np.zeros(nparams)  	# include norm
			
			# get model with lowest chi-square: adjusting flux uncertainties, accounting for systematic error 
			idx0, min_val = min(enumerate(mod_chi2), key=itemgetter(1))

			# best-fit model SED
			fluxes = np.zeros(nbands)
			for bb in range(0,nbands):
				fluxes[bb] = data_randmod[filters[bb]][idx0]

			print ("Reduced chi2 of best-fit model SED: %lf" % (mod_chi2[idx0]/nbands))
			if mod_chi2[idx0]/nbands > redcd_chi2:  
				sys_err_frac = 0.01
				while sys_err_frac <= 0.5:
					# modifiy flux uncertainties:
					modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
					chi2 = calc_modchi2_leastnorm(obs_fluxes,modif_obs_flux_err,fluxes)
					if chi2/nbands <= redcd_chi2:
						break
					sys_err_frac = sys_err_frac + 0.01
				print ("After adding %lf fraction to systematic error, reduced chi2 of best-fit model becomes: %lf" % (sys_err_frac,chi2/nbands))

			elif mod_chi2[idx0]/nbands <= redcd_chi2:
				# parameters of best-fit model
				for pp in range(0,nparams-1):
					params_chi2min_initfit[pp] = data_randmod[params[pp]][idx0]

				# model_leastnorm outputs norm in linear scale, not logarithmic
				norm_temp = model_leastnorm(obs_fluxes,obs_flux_err,fluxes) 
				params_chi2min_initfit[int(nparams)-1] = np.log10(norm_temp)
				print ("=> Parameters of best-fit  model SED: ")
				print (params_chi2min_initfit)

		elif rank>0:
			modif_obs_flux_err = np.zeros(nbands)
			params_chi2min_initfit = np.zeros(nparams)  # include log_norm

		comm.Bcast(modif_obs_flux_err, root=0)
		comm.Bcast(params_chi2min_initfit, root=0)

	elif increase_ferr == 0:
		idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
		# allocate memory in each process
		recvbuf_idx = np.empty(numDataPerRank, dtype='d')       # allocate space for recvbuf
		# scatter the ids to the processes
		comm.Scatter(idx_mpi, recvbuf_idx, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		count = 0
		for ii in recvbuf_idx:
			fluxes = np.zeros(nbands)
			for bb in range(0,nbands):
				fluxes[bb] = data_randmod[filters[bb]][int(ii)]

			chi2 = calc_modchi2_leastnorm(obs_fluxes,obs_flux_err,fluxes) 
			mod_chi2_temp[int(count)] = chi2

			count = count + 1

		mod_chi2 = None
		if rank == 0:
			mod_chi2 = np.empty(numDataPerRank*size, dtype='d')
				
		# gather the scattered data
		comm.Gather(mod_chi2_temp, mod_chi2, root=0)

		if rank == 0:
			# parameters of best-fit model
			idx0, min_val = min(enumerate(mod_chi2), key=itemgetter(1))
			# parameters of best-fit model
			params_chi2min_initfit = np.zeros(nparams)  	# include log_norm
			for pp in range(0,nparams-1):
				params_chi2min_initfit[pp] = data_randmod[params[pp]][idx0]
			fluxes = np.zeros(nbands)
			for bb in range(0,nbands):
				fluxes[bb] = data_randmod[filters[bb]][idx0]
			norm_temp = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
			params_chi2min_initfit[int(nparams)-1] = np.log10(norm_temp)

		elif rank>0:
			params_chi2min_initfit = np.zeros(nparams) 		# include norm
			
		comm.Bcast(params_chi2min_initfit, root=0)

		# empty arrays
		modif_obs_flux_err = np.zeros(nbands)

	return params_chi2min_initfit,modif_obs_flux_err


## prior function:
def lnprior(theta):
	par_val = np.zeros(nparams)
	if nparams == 5:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4] = theta
	elif nparams == 6:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5] = theta
	elif nparams == 7:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6] = theta
	elif nparams == 8:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7] = theta
	elif nparams == 9:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8] = theta
	elif nparams == 10:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8], par_val[9] = theta
	elif nparams == 11:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8], par_val[9], par_val[10] = theta
	elif nparams == 12:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8], par_val[9], par_val[10], par_val[11] = theta
	elif nparams == 13:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8], par_val[9], par_val[10], par_val[11], par_val[12] = theta
	elif nparams == 14:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8], par_val[9], par_val[10], par_val[11], par_val[12], par_val[13] = theta

	pass_flag = 0
	for pp in range(0,nparams):
		if priors_min[pp] <= par_val[pp] <= priors_max[pp]:
			pass_flag = pass_flag + 1
	if pass_flag == nparams:
		return 0.0
	return -np.inf
 

## function to calculate posterior probability:
def lnprob(theta):
	global obs_fluxes, obs_flux_err

	par_val = np.zeros(nparams)
	if nparams == 5:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4] = theta
	elif nparams == 6:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5] = theta
	elif nparams == 7:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6] = theta
	elif nparams == 8:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7] = theta
	elif nparams == 9:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8] = theta
	elif nparams == 10:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8], par_val[9] = theta
	elif nparams == 11:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8], par_val[9], par_val[10] = theta
	elif nparams == 12:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8], par_val[9], par_val[10], par_val[11] = theta
	elif nparams == 13:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8], par_val[9], par_val[10], par_val[11], par_val[12] = theta
	elif nparams == 14:
		par_val[0], par_val[1], par_val[2], par_val[3], par_val[4], par_val[5], par_val[6], par_val[7], par_val[8], par_val[9], par_val[10], par_val[11], par_val[12], par_val[13] = theta

	params_val = def_params_val
	for pp in range(0,nparams):     	# including log_mass
		params_val[params[pp]] = par_val[pp]

	# model fluxes
	photo_SED = generate_modelSED_photo_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,filters=filters,add_igm_absorption=add_igm_absorption,
											igm_type=igm_type,params_fsps=params_fsps, params_val=params_val,DL_Gpc=DL_Gpc,cosmo=cosmo_str,H0=H0,Om0=Om0,
											free_z=free_z,trans_fltr_int=trans_fltr_int)

	mod_fluxes = photo_SED['flux']

	if gauss_likelihood_form == 0:
		ln_likeli = gauss_ln_prob(obs_fluxes,obs_flux_err,mod_fluxes)
	elif gauss_likelihood_form == 1:
		chi2 = calc_chi2(obs_fluxes,obs_flux_err,mod_fluxes)
		ln_likeli = -0.5*chi2

	# get prior:
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + ln_likeli

"""
USAGE: mpirun -np [npros] python ./mcmc_pcmod_p1.py (1)filters (2)conf (3)inputSED_txt (4)name_params_list 
													(5)name_sampler_list (6)name_modif_obs_photo_SED
"""

temp_dir = PIXEDFIT_HOME+'/data/temp/'

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# get configuration file
global config_data
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
# nebular emission switch
global add_neb_emission
add_neb_emission = int(config_data['add_neb_emission'])
# gas_logu
global gas_logu
gas_logu = float(config_data['gas_logu'])

# SFH form
global sfh_form
if int(config_data['sfh_form']) == 0:
	sfh_form = 'tau_sfh'
elif int(config_data['sfh_form']) == 1:
	sfh_form = 'delayed_tau_sfh'
elif int(config_data['sfh_form']) == 2:
	sfh_form = 'log_normal_sfh'
elif int(config_data['sfh_form']) == 3:
	sfh_form = 'gaussian_sfh'
elif int(config_data['sfh_form']) == 4:
	sfh_form = 'double_power_sfh'

# galaxy's redshift
global gal_z
gal_z = float(config_data['gal_z'])

# input SED
global obs_fluxes, obs_flux_err
inputSED_txt = str(sys.argv[3])
data = np.loadtxt(temp_dir+inputSED_txt)
obs_fluxes = np.asarray(data[:,0])
obs_flux_err = np.asarray(data[:,1])

# IMF
global imf
imf = int(config_data['imf_type'])
# switch of dust emission modeling
global duste_switch
if int(config_data['duste_switch']) == 0:
	duste_switch = 'noduste'
elif int(config_data['duste_switch']) == 1:
	duste_switch = 'duste'
# dust extintion law 
global dust_ext_law
if int(config_data['dust_ext_law']) == 0:
	dust_ext_law = 'CF2000'
elif int(config_data['dust_ext_law']) == 1:
	dust_ext_law = 'Cal2000'
# switch of igm absorption
global add_igm_absorption, igm_type
add_igm_absorption = int(config_data['add_igm_absorption'])
igm_type = int(config_data['igm_type'])
# number of walkers, steps, and nsteps_cut
global nwalkers, nsteps, nsteps_cut 
nwalkers = int(config_data['nwalkers'])
nsteps = int(config_data['nsteps'])
nsteps_cut = int(config_data['nsteps_cut'])
# original number of processors
global ori_nproc
ori_nproc = int(config_data['ori_nproc'])
# whether dust_index is fix or not
global fix_dust_index, fix_dust_index_val
if float(config_data['pr_dust_index_min']) == float(config_data['pr_dust_index_max']):     ### dust_index is set fix
	fix_dust_index = 1
	fix_dust_index_val = float(config_data['pr_dust_index_min'])
elif float(config_data['pr_dust_index_min']) != float(config_data['pr_dust_index_max']):   ### dust_index is varies
	fix_dust_index = 0
	fix_dust_index_val = 0
# switch of AGN dusty torus emission
global add_agn 
add_agn = int(config_data['add_agn'])

global gauss_likelihood_form
gauss_likelihood_form = int(config_data['gauss_likelihood_form'])

# name of fits file containg pre-calculated random model SEDs
name_saved_randmod = config_data['name_saved_randmod']

# cosmology
# The choices are: [0:flat_LCDM, 1:WMAP5, 2:WMAP7, 3:WMAP9, 4:Planck13, 5:Planck15]
global cosmo_str, H0, Om0
cosmo = int(config_data['cosmo'])
if cosmo==0: 
	cosmo_str = 'flat_LCDM' 
elif cosmo==1:
	cosmo_str = 'WMAP5'
elif cosmo==2:
	cosmo_str = 'WMAP7'
elif cosmo==3:
	cosmo_str = 'WMAP9'
elif cosmo==4:
	cosmo_str = 'Planck13'
elif cosmo==5:
	cosmo_str = 'Planck15'
#elif cosmo==6:
#	cosmo_str = 'Planck18'
else:
	print ("The cosmo input is not recognized!")
	sys.exit()
H0 = float(config_data['H0'])
Om0 = float(config_data['Om0'])


global free_z, DL_Gpc
if gal_z<=0:
	free_z = 1
	DL_Gpc = 0
else:
	free_z = 0
	# luminosity distance
	if cosmo_str=='flat_LCDM':
		cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
		DL_Gpc0 = cosmo1.luminosity_distance(gal_z)      # in unit of Mpc
	elif cosmo_str=='WMAP5':
		DL_Gpc0 = WMAP5.luminosity_distance(gal_z)
	elif cosmo_str=='WMAP7':
		DL_Gpc0 = WMAP7.luminosity_distance(gal_z)
	elif cosmo_str=='WMAP9':
		DL_Gpc0 = WMAP9.luminosity_distance(gal_z)
	elif cosmo_str=='Planck13':
		DL_Gpc0 = Planck13.luminosity_distance(gal_z)
	elif cosmo_str=='Planck15':
		DL_Gpc0 = Planck15.luminosity_distance(gal_z)
	#elif cosmo_str=='Planck18':
	#	DL_Gpc0 = Planck18.luminosity_distance(gl_z)
	DL_Gpc = DL_Gpc0.value/1.0e+3

# default parameter set
global def_params, def_params_val
def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta','log_age','dust_index','dust1','dust2',
				'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

#def_params_val={'log_mass':0.0,'z':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,'log_qpah':-99.0,'log_umin':-99.0,
#					'log_gamma':-99.0,'dust1':-99.0,'dust2':-99.0,'dust_index':-99.0,'log_age':-99.0,
#					'log_alpha':-99.0, 'log_beta':-99.0, 'log_t0':-99.0,'log_tau':-99.0,'logzsol':-99.0}

def_params_val = {'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

if fix_dust_index == 1:
	def_params_val['dust_index'] = fix_dust_index_val
if fix_dust_index == 0:
	def_params_val['dust_index'] = -0.7
if free_z == 0:
	def_params_val['z'] = gal_z

global def_params_fsps, params_assoc_fsps, status_log
def_params_fsps = ['logzsol', 'log_tau', 'log_age', 'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 
				'log_qpah','log_fagn', 'log_tauagn']
params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

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

# get number of parameters:
global params, nparams
params, nparams = get_params(free_z, sfh_form, duste_switch, dust_ext_law, add_agn, fix_dust_index)
if rank == 0:
	print ("parameters: ")
	print (params)
	print ("number of parameters: %d" % nparams)

global params_fsps, nparams_fsps
params_fsps = []
for ii in range(0,len(def_params_fsps)):
	for jj in range(0,nparams):
		if def_params_fsps[ii] == params[jj]:
			params_fsps.append(params[jj])
nparams_fsps = len(params_fsps)

# priors range for the parameters: including normalization
global priors_min, priors_max
priors_min = np.zeros(nparams)
priors_max = np.zeros(nparams)
for ii in range(0,nparams-1):   # without log_mass       
	str_temp = 'pr_%s_min' % params[ii]
	priors_min[ii] = float(config_data[str_temp])
	str_temp = 'pr_%s_max' % params[ii]
	priors_max[ii] = float(config_data[str_temp])

# if redshift is fix: free_z=0, 
# match wavelength points of transmission curves with that of spectrum at a given redshift
global trans_fltr_int
if free_z == 0:
	params_val = def_params_val
	for pp in range(0,nparams-1):  # without log_mass
		params_val[params[pp]] = priors_min[pp]

	spec_SED = generate_modelSED_spec_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,add_igm_absorption=add_igm_absorption,
														igm_type=igm_type,params_fsps=params_fsps, params_val=params_val,DL_Gpc=DL_Gpc)

	nwave = len(spec_SED['wave'])
	if rank == 0:
		trans_fltr_int = match_filters_array(spec_SED['wave'],filters)
	elif rank>0:
		trans_fltr_int = np.zeros((nbands,nwave))

	comm.Bcast(trans_fltr_int, root=0)


#####$$$$$ Initial fitting $$$$$$######
# get number of model SEDs
global npmod_seds
hdu = fits.open(name_saved_randmod)
npmod_seds0 = int(hdu[0].header['nrows'])
hdu.close()
npmod_seds = int(npmod_seds0/size)*size

# mod_fluxes0 = [idx-band][idx-model]  --> only relevant if increase_ferr==0
# mod_params1 = [idx-param][idx-model]  --> only relevant if increase_ferr==0
#print ("Initial fitting")
increase_ferr = 1
params_chi2min_initfit,modif_obs_flux_err = initfit(increase_ferr,name_saved_randmod)
if np.sum(modif_obs_flux_err) != 0:
	obs_flux_err = modif_obs_flux_err
	increase_ferr = 0
	params_chi2min_initfit,modif_obs_flux_err = initfit(increase_ferr,name_saved_randmod)
if rank == 0:
	print ("Best-fit parameters from initial fitting:")
	print (params_chi2min_initfit)

#####$$$$$ MCMC SED fitting $$$$$$#########
# add priors for normalization:
priors_min[int(nparams)-1] = params_chi2min_initfit[int(nparams)-1] - 2.0 
priors_max[int(nparams)-1] = params_chi2min_initfit[int(nparams)-1] + 2.0

init_pos = params_chi2min_initfit
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

	name_temp = str(sys.argv[4])
	file_out = open(name_temp,"w")
	for pp in range(0,nparams):
		file_out.write("%s\n" % params[pp])
	file_out.close()
	
	name_temp = str(sys.argv[5])
	file_out = open(name_temp,"w")
	nsamples = samples.shape[0]
	for ii in range(0,nsamples):
		for pp in range(0,nparams-1):
			file_out.write("%e " % samples[ii][pp])
		file_out.write("%e \n" % samples[ii][int(nparams)-1])
	file_out.close()

	name_temp = str(sys.argv[6])
	file_out = open(name_temp,"w")
	for bb in range(0,nbands):
		file_out.write("%e  %e\n" % (obs_fluxes[bb],obs_flux_err[bb]))
	file_out.close()



