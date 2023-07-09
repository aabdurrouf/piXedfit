import numpy as np 
from math import log10, pow
import sys, os 
import fsps
import h5py
from mpi4py import MPI
from astropy.io import fits
from astropy.cosmology import *

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.piXedfit_model import * 

## USAGE: mpirun -np [npros] python ./save_models_spec_h5.py (1)name_config

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

config_file = str(sys.argv[1])
dir_file = PIXEDFIT_HOME+'/data/temp/'

imf, add_neb_emission, add_igm_absorption, igm_type, sfh_form, dust_law, duste_switch, add_agn, nmodels, free_gas_logz, smooth_velocity, sigma_smooth, smooth_lsf, name_file_lsf, name_out, params, nparams, priors_min, priors_max, cosmo_str, cosmo, H0, Om0, gal_z = read_config_file_genmodels(dir_file,config_file)

if rank == 0:
	print ("There are %d parameters: " % nparams)
	print (params)

# normalize with size
nmodels = int(nmodels/size)*size

params_fsps, nparams_fsps = get_params_fsps(params)
def_params_val = default_params_val()

# check whether smoothing with a line spread function or not
if smooth_lsf == True or smooth_lsf == 1:
	data = np.loadtxt(dir_file+name_file_lsf)
	lsf_wave, lsf_sigma = data[:,0], data[:,1]
else:
	lsf_wave, lsf_sigma = None, None

# call FSPS
global sp 
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)
sp = set_initial_fsps(sp,duste_switch,add_neb_emission,add_agn,sfh_form,dust_law,smooth_velocity=smooth_velocity,sigma_smooth=sigma_smooth,
					smooth_lsf=smooth_lsf,lsf_wave=lsf_wave,lsf_sigma=lsf_sigma)

# generate random parameters
rand_params = np.zeros((nparams,nmodels))
for pp in range(nparams):
	rand_params[pp] = np.random.uniform(priors_min[pp],priors_max[pp],nmodels)
	
#for pp in range(nparams):
#	if params[pp] == 'logzsol':
#		rand_params[pp] = np.random.normal(-0.3,0.3,nmodels)
#	else:
#		rand_params[pp] = np.random.uniform(priors_min[pp],priors_max[pp],nmodels)

# scattering the multiple processes
numDataPerRank = int(nmodels/size)
idx_mpi = np.linspace(0,nmodels-1,nmodels)
recvbuf_idx = np.empty(numDataPerRank, dtype='d')
comm.Scatter(idx_mpi, recvbuf_idx, root=0)

# get number of wavelength grids
params_val = def_params_val
for pp in range(0,nparams):  # log_mass is excluded
	params_val[params[pp]] = priors_min[pp]
wave, spec_flux, formed_mass, SFR_fSM, dust_mass, log_fagn_bol, mw_age = generate_modelSED_spec_restframe_props(sp=sp,imf_type=imf,sfh_form=sfh_form,
																				add_agn=add_agn,params_fsps=params_fsps, params_val=params_val)
nwaves = len(wave)

mod_params_temp = np.zeros((nparams,numDataPerRank))
mod_log_mass_temp = np.zeros(numDataPerRank)
mod_log_sfr_temp = np.zeros(numDataPerRank)
mod_log_mw_age_temp = np.zeros(numDataPerRank)
if duste_switch==1:
	mod_log_dustmass_temp = np.zeros(numDataPerRank)
if add_agn == 1:
	mod_log_fagn_bol_temp = np.zeros(numDataPerRank)
mod_fluxes_temp = np.zeros((nwaves,numDataPerRank))

# wavelength
mod_spec_wave = np.zeros(nwaves) 

count = 0
for ii in recvbuf_idx:
	params_val = def_params_val
	for pp in range(0,nparams):
		temp_val = rand_params[pp][int(ii)] 
		mod_params_temp[pp][int(count)] = temp_val
		params_val[params[pp]] = temp_val

	wave, spec_flux, formed_mass, SFR_fSM, dust_mass, log_fagn_bol, mw_age = generate_modelSED_spec_restframe_props(sp=sp,imf_type=imf,sfh_form=sfh_form,
																				  add_agn=add_agn,params_fsps=params_fsps, params_val=params_val)
	mod_log_mass_temp[int(count)] = log10(formed_mass)
	mod_log_sfr_temp[int(count)] = log10(SFR_fSM)
	mod_log_mw_age_temp[int(count)] = np.log10(mw_age)

	if duste_switch==1:
		mod_log_dustmass_temp[int(count)] = log10(dust_mass)  

	if add_agn == 1:
		mod_log_fagn_bol_temp[int(count)] = log_fagn_bol

	mod_fluxes_temp[:,int(count)] = spec_flux
	mod_spec_wave = wave

	count = count + 1

	sys.stdout.write('\r')
	sys.stdout.write('rank: %d  Calculation process: %d from %d --> %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
	sys.stdout.flush()
sys.stdout.write('\n')

# gather the scattered data
mod_log_mass = np.zeros(nmodels)
mod_log_sfr = np.zeros(nmodels)
mod_log_mw_age = np.zeros(nmodels)
if duste_switch==1:
	mod_log_dustmass = np.zeros(nmodels)
if add_agn == 1:
	mod_log_fagn_bol = np.zeros(nmodels)

mod_params = np.zeros((nparams,nmodels))
mod_fluxes = np.zeros((nwaves,nmodels))

comm.Gather(mod_log_mass_temp, mod_log_mass, root=0)
comm.Gather(mod_log_sfr_temp, mod_log_sfr, root=0)
comm.Gather(mod_log_mw_age_temp, mod_log_mw_age, root=0)
if duste_switch==1:
	comm.Gather(mod_log_dustmass_temp, mod_log_dustmass, root=0)
if add_agn == 1:
	comm.Gather(mod_log_fagn_bol_temp, mod_log_fagn_bol, root=0)

for pp in range(0,nparams):
	comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

for bb in range(0,nwaves):
	comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)

# transpose from (wave,id) -> (id,wave)
mod_fluxes_trans = np.transpose(mod_fluxes, axes=(1,0))

# store into HDF5 file
if rank == 0:
	with h5py.File(name_out, 'w') as f:
		m = f.create_group('mod')

		# info
		m.attrs['imf_type'] = imf 
		m.attrs['sfh_form'] = sfh_form
		m.attrs['dust_law'] = dust_law
		m.attrs['duste_switch'] = duste_switch
		m.attrs['add_neb_emission'] = add_neb_emission
		m.attrs['add_agn'] = add_agn
		m.attrs['free_gas_logz'] = free_gas_logz
		m.attrs['smooth_velocity'] = smooth_velocity
		m.attrs['sigma_smooth'] = sigma_smooth
		m.attrs['smooth_lsf'] = smooth_lsf
		if smooth_lsf == True or smooth_lsf == 1:
			m.attrs['name_file_lsf'] = name_file_lsf
		m.attrs['funit'] = 'L_sun/A'
		m.attrs['nmodels'] = nmodels
		m.attrs['nparams'] = nparams
		for pp in range(0,nparams):
			str_temp = 'par%d' % pp
			m.attrs[str_temp] = params[pp]

		add_par = 0
		str_temp = 'par%d' % (nparams+add_par)
		m.attrs[str_temp] = 'log_mass'

		add_par = add_par + 1
		str_temp = 'par%d' % (nparams+add_par)
		m.attrs[str_temp] = 'log_sfr'

		add_par = add_par + 1
		str_temp = 'par%d' % (nparams+add_par)
		m.attrs[str_temp] = 'log_mw_age'

		if duste_switch==1:
			add_par = add_par + 1
			str_temp = 'par%d' % (nparams+add_par)
			m.attrs[str_temp] = 'log_dustmass'

		if add_agn == 1:
			add_par = add_par + 1
			str_temp = 'par%d' % (nparams+add_par)
			m.attrs[str_temp] = 'log_fagn_bol'

		m.attrs['nparams_all'] = int(nparams+add_par+1)

		for pp in range(0,nparams):
			str_temp = 'pr_%s_min' % params[pp]
			m.attrs[str_temp] = priors_min[pp]
			str_temp = 'pr_%s_max' % params[pp]
			m.attrs[str_temp] = priors_max[pp]

		# parameters
		p = m.create_group('par')
		for pp in range(0,int(nparams)):
			p.create_dataset(params[pp], data=np.array(mod_params[pp]), compression="gzip")

		p.create_dataset('log_mass', data=np.array(mod_log_mass), compression="gzip")
		p.create_dataset('log_sfr', data=np.array(mod_log_sfr), compression="gzip")
		p.create_dataset('log_mw_age', data=np.array(mod_log_mw_age), compression="gzip")

		if duste_switch==1:
			p.create_dataset('log_dustmass', data=np.array(mod_log_dustmass), compression="gzip")

		if add_agn == 1:
			p.create_dataset('log_fagn_bol', data=np.array(mod_log_fagn_bol), compression="gzip") 

		# spectra
		s = m.create_group('spec')

		s.create_dataset('wave', data=np.array(mod_spec_wave), compression="gzip") 
		for ii in range(0,nmodels):
			str_temp = 'f%d' % ii
			s.create_dataset(str_temp, data=np.array(mod_fluxes_trans[ii]), compression="gzip")

	sys.stdout.write('\n')

