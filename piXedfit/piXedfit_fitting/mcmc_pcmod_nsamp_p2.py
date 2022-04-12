import numpy as np
from math import log10, pow
import sys, os
import fsps
import emcee
from mpi4py import MPI
from astropy.io import fits

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)


from piXedfit.piXedfit_model import calc_mw_age, get_dust_mass_mainSFH_fit, get_dust_mass_fagnbol_mainSFH_fit, get_dust_mass_othSFH_fit, get_sfr_dust_mass_othSFH_fit, get_sfr_dust_mass_fagnbol_othSFH_fit, construct_SFH


# Function to store the sampler chains into output fits file:
def store_to_fits(sampler_params=None,sampler_log_sfr=None,sampler_log_mw_age=None,
	sampler_logdustmass=None,sampler_log_fagn_bol=None,fits_name_out=None): 

	hdr = fits.Header()
	hdr['imf'] = imf
	hdr['nparams'] = nparams
	hdr['sfh_form'] = sfh_form
	hdr['dust_ext_law'] = dust_ext_law
	hdr['nfilters'] = nbands
	hdr['duste_stat'] = duste_switch
	hdr['add_neb_emission'] = add_neb_emission
	if add_neb_emission == 1:
		hdr['gas_logu'] = gas_logu
	hdr['add_agn'] = add_agn
	hdr['add_igm_absorption'] = add_igm_absorption
	hdr['cosmo'] = cosmo_str
	hdr['H0'] = H0
	hdr['Om0'] = Om0
	if duste_switch == 'duste':
		if fix_dust_index == 1:
			hdr['dust_index'] = fix_dust_index_val
	if add_igm_absorption == 1:
		hdr['igm_type'] = igm_type
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		hdr[str_temp] = filters[bb]

		str_temp = 'flux%d' % bb
		if np.isnan(obs_fluxes[bb])==False:
			hdr[str_temp] = obs_fluxes[bb]
		elif np.isnan(obs_fluxes[bb])==True:
			hdr[str_temp] = -99.0

		str_temp = 'flux_err%d' % bb 
		if np.isnan(obs_flux_err[bb])==False:
			hdr[str_temp] = obs_flux_err[bb]
		elif np.isnan(obs_flux_err[bb])==True:
			hdr[str_temp] = -99.0

	if free_z == 0:
		hdr['gal_z'] = gal_z
		hdr['free_z'] = 0
	elif free_z == 1:
		hdr['free_z'] = 1
	# add parameters
	for pp in range(0,nparams):
		str_temp = 'param%d' % pp
		hdr[str_temp] = params[pp]

	# add columns
	cols0 = []
	col_count = 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'rows'
	col = fits.Column(name='rows', format='3A', array=['p16','p50','p84'])
	cols0.append(col)

	#=> basic params
	for pp in range(0,nparams):
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = params[pp]
		p16 = np.percentile(sampler_params[params[pp]],16)
		p50 = np.percentile(sampler_params[params[pp]],50)
		p84 = np.percentile(sampler_params[params[pp]],84)
		col = fits.Column(name=params[pp], format='D', array=np.array([p16,p50,p84]))
		cols0.append(col)

	#=> SFR
	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_sfr'
	p16 = np.percentile(sampler_log_sfr,16)
	p50 = np.percentile(sampler_log_sfr,50)
	p84 = np.percentile(sampler_log_sfr,84)
	col = fits.Column(name='log_sfr', format='D', array=np.array([p16,p50,p84]))
	cols0.append(col)

	#=> mass-weighted age
	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_mw_age'
	p16 = np.percentile(sampler_log_mw_age,16)
	p50 = np.percentile(sampler_log_mw_age,50)
	p84 = np.percentile(sampler_log_mw_age,84)
	col = fits.Column(name='log_mw_age', format='D', array=np.array([p16,p50,p84]))
	cols0.append(col)

	#=> dust mass
	if duste_switch == 'duste':
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = 'log_dustmass'
		p16 = np.percentile(sampler_logdustmass,16)
		p50 = np.percentile(sampler_logdustmass,50)
		p84 = np.percentile(sampler_logdustmass,84)
		col = fits.Column(name='log_dustmass', format='D', array=np.array([p16,p50,p84]))
		cols0.append(col)

	#=> AGN
	if add_agn == 1:
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = 'log_fagn_bol'
		p16 = np.percentile(sampler_log_fagn_bol,16)
		p50 = np.percentile(sampler_log_fagn_bol,50)
		p84 = np.percentile(sampler_log_fagn_bol,84)
		col = fits.Column(name='log_fagn_bol', format='D', array=np.array([p16,p50,p84]))
		cols0.append(col)

	hdr['ncols'] = col_count

	# combine all
	primary_hdu = fits.PrimaryHDU(header=hdr)
	cols = fits.ColDefs(cols0)
	hdu = fits.BinTableHDU.from_columns(cols)

	hdul = fits.HDUList([primary_hdu, hdu])
	hdul.writeto(fits_name_out, overwrite=True)


def calc_sampler_mwage(nsamples=0,sampler_pop_mass=[],sampler_tau=[],sampler_t0=[],
	sampler_alpha=[],sampler_beta=[],sampler_age=[]):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_mw_age0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		pop_mass = sampler_pop_mass[int(ii)]
		tau = sampler_tau[int(ii)]
		age = sampler_age[int(ii)]
		if sfh_form == 'tau_sfh' or sfh_form == 'delayed_tau_sfh':
			sampler_mw_age0[int(count)] = calc_mw_age(sfh_form=sfh_form,tau=tau,age=age,
																formed_mass=pop_mass)
		elif sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
			t0 = sampler_t0[int(ii)]
			sampler_mw_age0[int(count)] = calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,age=age,
																formed_mass=pop_mass)
		elif sfh_form == 'double_power_sfh':
			alpha = sampler_alpha[int(ii)]
			beta = sampler_beta[int(ii)]
			sampler_mw_age0[int(count)] = calc_mw_age(sfh_form=sfh_form,tau=tau,alpha=alpha,beta=beta,
															age=age,formed_mass=pop_mass)

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	sys.stdout.write('\n')

	sampler_mw_age = np.zeros(numDataPerRank*size)
	comm.Gather(sampler_mw_age0, sampler_mw_age, root=0)
	comm.Bcast(sampler_mw_age, root=0)

	sampler_log_mw_age = np.log10(sampler_mw_age)
	return sampler_log_mw_age


def calc_sampler_dustmass_fagnbol_mainSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_logdustmass0 = np.zeros(numDataPerRank)
	sampler_log_fagn_bol0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = sampler_params[params[pp]][int(ii)] 
		dust_mass, fagn_bol = get_dust_mass_fagnbol_mainSFH_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,params_fsps=params_fsps, params_val=params_val)

		if dust_mass <= 0:
			sampler_logdustmass0[int(count)] = -99.0
		elif dust_mass > 0:
			sampler_logdustmass0[int(count)] = log10(dust_mass)

		sampler_log_fagn_bol0[int(count)] = log10(fagn_bol) 

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	sys.stdout.write('\n')

	sampler_logdustmass = np.zeros(numDataPerRank*size)
	sampler_log_fagn_bol = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_logdustmass0, sampler_logdustmass, root=0)
	comm.Bcast(sampler_logdustmass, root=0)

	comm.Gather(sampler_log_fagn_bol0, sampler_log_fagn_bol, root=0)
	comm.Bcast(sampler_log_fagn_bol, root=0)

	return sampler_logdustmass, sampler_log_fagn_bol


def calc_sampler_dustmass_mainSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_logdustmass0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# dust-mass
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = sampler_params[params[pp]][int(ii)] 
		dust_mass = get_dust_mass_mainSFH_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,params_fsps=params_fsps, params_val=params_val)

		if dust_mass <= 0:
			sampler_logdustmass0[int(count)] = -99.0
		elif dust_mass > 0:
			sampler_logdustmass0[int(count)] = log10(dust_mass)

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	sys.stdout.write('\n')

	sampler_logdustmass = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_logdustmass0, sampler_logdustmass, root=0)
	comm.Bcast(sampler_logdustmass, root=0)
	return sampler_logdustmass

def calc_sampler_dustmass_othSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_logdustmass0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# dust-mass
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = sampler_params[params[pp]][int(ii)] 
		dust_mass = get_dust_mass_othSFH_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,params_fsps=params_fsps, params_val=params_val)

		if dust_mass <= 0:
			sampler_logdustmass0[int(count)] = -99.0
		elif dust_mass > 0:
			sampler_logdustmass0[int(count)] = log10(dust_mass)

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	sys.stdout.write('\n')

	sampler_logdustmass = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_logdustmass0, sampler_logdustmass, root=0)
	comm.Bcast(sampler_logdustmass, root=0)
	return sampler_logdustmass

def calc_sampler_SFR_dustmass_fagnbol_othSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_log_sfr0 = np.zeros(numDataPerRank)
	sampler_logdustmass0 = np.zeros(numDataPerRank)
	sampler_log_fagn_bol0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = sampler_params[params[pp]][int(ii)] 

		SFR, dust_mass, log_fagn_bol = get_sfr_dust_mass_fagnbol_othSFH_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,params_fsps=params_fsps, params_val=params_val)

		sampler_log_sfr0[int(count)] = log10(SFR)
		if dust_mass <= 0:
			sampler_logdustmass0[int(count)] = -99.0
		elif dust_mass > 0:
			sampler_logdustmass0[int(count)] = log10(dust_mass)

		sampler_log_fagn_bol0[int(count)] = log_fagn_bol

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	sys.stdout.write('\n')

	sampler_log_sfr = np.zeros(numDataPerRank*size)
	sampler_logdustmass = np.zeros(numDataPerRank*size)
	sampler_log_fagn_bol = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_log_sfr0, sampler_log_sfr, root=0)
	comm.Gather(sampler_logdustmass0, sampler_logdustmass, root=0)
	comm.Gather(sampler_log_fagn_bol0, sampler_log_fagn_bol, root=0)

	comm.Bcast(sampler_log_sfr, root=0)
	comm.Bcast(sampler_logdustmass, root=0)
	comm.Bcast(sampler_log_fagn_bol, root=0)

	return sampler_log_sfr, sampler_logdustmass, sampler_log_fagn_bol


def calc_sampler_SFR_dustmass_othSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_log_sfr0 = np.zeros(numDataPerRank)
	sampler_logdustmass0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		# dust-mass
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = sampler_params[params[pp]][int(ii)] 
		SFR,dust_mass = get_sfr_dust_mass_othSFH_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,params_fsps=params_fsps, params_val=params_val)
		sampler_log_sfr0[int(count)] = log10(SFR)
		
		if dust_mass <= 0:
			sampler_logdustmass0[int(count)] = -99.0
		elif dust_mass > 0:
			sampler_logdustmass0[int(count)] = log10(dust_mass)

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	sys.stdout.write('\n')

	sampler_log_sfr = np.zeros(numDataPerRank*size)
	sampler_logdustmass = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_log_sfr0, sampler_log_sfr, root=0)
	comm.Gather(sampler_logdustmass0, sampler_logdustmass, root=0)
	comm.Bcast(sampler_log_sfr, root=0)
	comm.Bcast(sampler_logdustmass, root=0)
	return (sampler_log_sfr,sampler_logdustmass)

def calc_sampler_SFR_othSFH(nsamples=0,sampler_params=None):
	numDataPerRank = int(nsamples/size)
	idx_mpi = np.linspace(0,nsamples-1,nsamples)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	sampler_log_sfr0 = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		tau = pow(10.0,sampler_params['log_tau'][int(ii)])
		age = pow(10.0,sampler_params['log_age'][int(ii)])
		formed_mass = pow(10.0,sampler_params['log_mass'][int(ii)])
		t0 = 0.0
		alpha = 0.0
		beta = 0.0

		if sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
			t0 = pow(10.0,sampler_params['log_t0'][int(ii)])
		if sfh_form == 'double_power_sfh':
			alpha = pow(10.0,sampler_params['log_alpha'][int(ii)])
			beta = pow(10.0,sampler_params['log_beta'][int(ii)])
		
		t, SFR_t = construct_SFH(sfh_form=sfh_form,t0=t0,tau=tau,alpha=alpha,beta=beta,age=age,formed_mass=formed_mass)

		idx_excld = np.where((SFR_t<=0) | (np.isnan(SFR_t)==True) | (np.isinf(SFR_t)==True))
		t = np.delete(t, idx_excld[0])
		SFR_t = np.delete(SFR_t, idx_excld[0])
		if len(t)>0:
			sampler_log_sfr0[int(count)] = log10(SFR_t[len(t)-1])
		
		#idx_sel = np.where(t>=age-0.01)		# SFR is average SFR grids within the last 10Myr
		
		#if len(idx_sel[0])>0:
		#	SFR_sel = [SFR_t[i] for i in idx_sel[0]]
		#	sampler_log_sfr0[int(count)] = math.log10(np.mean(SFR_sel))
		#else:
		#	sampler_log_sfr0[int(count)] = -99.0
		#sampler_log_sfr0[int(count)] = math.log10(SFR_t[len(t)-1])
		
		#if len(idx_sel[0])>0:
		#	SFR_sel = [SFR_t[i] for i in idx_sel[0]]
		#	sampler_log_sfr0[int(count)] = np.log10(np.mean(SFR_sel))
		#elif max(t)>age-0.5:
		#	sampler_log_sfr0[int(count)] = math.log10(SFR_t[len(t)-1])
		#else:
		#	sampler_log_sfr0[int(count)] = 1.0e-33

		count = count + 1
		sys.stdout.write('\r')
		sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		sys.stdout.flush()
	sys.stdout.write('\n')

	sampler_log_sfr = np.zeros(numDataPerRank*size)

	comm.Gather(sampler_log_sfr0, sampler_log_sfr, root=0)
	comm.Bcast(sampler_log_sfr, root=0)
	return sampler_log_sfr



"""
USAGE: mpirun -np [npros] python ./mcmc_pcmod_p2.py (1)filters_list (2)configuration file (3)name_params_list
													(4)name_sampler_list (5)name_modif_obs_photo_SED
													(6)name_out_finalfit
"""

temp_dir = PIXEDFIT_HOME+'/data/temp/'

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# configuration file
global config_data
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
# nebular emission
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
name_file = str(sys.argv[5])
data = np.loadtxt(temp_dir+name_file)
obs_fluxes = data[:,0]
obs_flux_err = data[:,1]

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
# whether dust_index is set fix or not
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

global free_z
if gal_z<=0:
	free_z = 1
else:
	free_z = 0

# default parameter set
global def_params, def_params_val
def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta','log_age','dust_index','dust1','dust2',
				'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

#def_params_val={'log_mass':0.0,'z':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,'log_qpah':-99.0,'log_umin':-99.0,
#					'log_gamma':-99.0,'dust1':-99.0,'dust2':-99.0,'dust_index':-99.0,'log_age':-99.0,
#					'log_alpha':-99.0, 'log_beta':-99.0, 'log_t0':-99.0,'log_tau':-99.0,'logzsol':-99.0}

def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
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
# dust emission switch
if duste_switch == 'duste':
	sp.params["add_dust_emission"] = True
elif duste_switch == 'noduste':
	sp.params["add_dust_emission"] = False
# nebular emission switch
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

# parameters
global params, nparams
name_temp = str(sys.argv[3])
params = np.genfromtxt(temp_dir+name_temp, dtype='str')
nparams = len(params)

global params_fsps, nparams_fsps
params_fsps = []
for ii in range(0,len(def_params_fsps)):
	for jj in range(0,nparams):
		if def_params_fsps[ii] == params[jj]:
			params_fsps.append(params[jj])
nparams_fsps = len(params_fsps)

# sampler chains
name_temp = str(sys.argv[4])
samplers_data = np.loadtxt(temp_dir+name_temp)
nsamples = len(samplers_data[:,0])

# allocate memory
sampler_params = {}
for pp in range(0,nparams):
	sampler_params[params[pp]] = np.zeros(nsamples)

for pp in range(0,nparams):
	rr = 0
	for ff in samplers_data[:,int(pp)]:
		sampler_params[params[pp]][rr] = float(ff)
		rr = rr + 1

sampler_pop_mass = np.power(10.0,sampler_params['log_mass'])
sampler_age = np.power(10.0,sampler_params['log_age'])
sampler_tau = np.power(10.0,sampler_params['log_tau'])

# calculate SFR, mw-age, dust mass, log_fagn_bol
sampler_log_sfr = None
sampler_log_mw_age = None
sampler_logdustmass = None
sampler_log_fagn_bol = None
if sfh_form == 'tau_sfh' or sfh_form == 'delayed_tau_sfh':
	# SFR
	sampler_SFR_exp = 1.0/np.exp(sampler_age/sampler_tau)
	if sfh_form == 'tau_sfh':
		sampler_log_sfr = np.log10(sampler_pop_mass*sampler_SFR_exp/sampler_tau/(1.0-sampler_SFR_exp)/1e+9)
	if sfh_form == 'delayed_tau_sfh':
		sampler_log_sfr = np.log10(sampler_pop_mass*sampler_age*sampler_SFR_exp/((sampler_tau*sampler_tau)-((sampler_age*sampler_tau)+(sampler_tau*sampler_tau))*sampler_SFR_exp)/1e+9)
	# MW-age
	sampler_log_mw_age = calc_sampler_mwage(nsamples=nsamples,sampler_pop_mass=sampler_pop_mass,sampler_tau=sampler_tau,sampler_age=sampler_age)
	# dust-mass
	if duste_switch == 'duste':
		if add_agn == 1:
			sampler_logdustmass, sampler_log_fagn_bol = calc_sampler_dustmass_fagnbol_mainSFH(nsamples=nsamples,sampler_params=sampler_params)
		elif add_agn == 0:
			sampler_logdustmass = calc_sampler_dustmass_mainSFH(nsamples=nsamples,sampler_params=sampler_params)

elif sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
	sampler_t0 = np.power(10.0,sampler_params['log_t0'])
	# MW-age
	sampler_log_mw_age = calc_sampler_mwage(nsamples=nsamples,sampler_pop_mass=sampler_pop_mass,sampler_tau=sampler_tau,sampler_t0=sampler_t0,sampler_age=sampler_age)
	# SFR and dust-mass
	if duste_switch == 'duste':
		if add_agn == 1:
			sampler_log_sfr, sampler_logdustmass, sampler_log_fagn_bol = calc_sampler_SFR_dustmass_fagnbol_othSFH(nsamples=nsamples,sampler_params=sampler_params)
		elif add_agn == 0:
			sampler_log_sfr, sampler_logdustmass = calc_sampler_SFR_dustmass_othSFH(nsamples=nsamples,sampler_params=sampler_params)
	else:
		sampler_log_sfr = calc_sampler_SFR_othSFH(nsamples=nsamples,sampler_params=sampler_params)


elif sfh_form == 'double_power_sfh':
	sampler_alpha = np.power(10.0,sampler_params['log_alpha'])
	sampler_beta = np.power(10.0,sampler_params['log_beta'])
	# MW-age
	sampler_log_mw_age = calc_sampler_mwage(nsamples=nsamples,sampler_pop_mass=sampler_pop_mass,sampler_tau=sampler_tau,
													sampler_alpha=sampler_alpha, sampler_beta=sampler_beta, sampler_age=sampler_age)
	# SFR and dust-mass
	if duste_switch == 'duste':
		if add_agn == 1:
			sampler_log_sfr, sampler_logdustmass, sampler_log_fagn_bol = calc_sampler_SFR_dustmass_fagnbol_othSFH(nsamples=nsamples,sampler_params=sampler_params)
		elif add_agn == 0:
			sampler_log_sfr, sampler_logdustmass = calc_sampler_SFR_dustmass_othSFH(nsamples=nsamples,sampler_params=sampler_params)
	else:
		sampler_log_sfr = calc_sampler_SFR_othSFH(nsamples=nsamples,sampler_params=sampler_params)

fits_name_out = str(sys.argv[6])
store_to_fits(sampler_params=sampler_params,sampler_log_sfr=sampler_log_sfr,
					sampler_log_mw_age=sampler_log_mw_age,sampler_logdustmass=sampler_logdustmass,
					sampler_log_fagn_bol=sampler_log_fagn_bol,fits_name_out=fits_name_out)

