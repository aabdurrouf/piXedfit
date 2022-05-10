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

from piXedfit.utils.filtering import match_filters_array
from piXedfit.piXedfit_model import generate_modelSED_spec_fit, generate_modelSED_propphoto_nomwage_fit, calc_mw_age, generate_modelSED_spec_restframe_props 
from piXedfit.piXedfit_fitting import get_params

## USAGE: mpirun -np [npros] python ./save_models_spec_h5.py (1)name_config

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# configuration file
global config_data
config_file = str(sys.argv[1])
dir_file = PIXEDFIT_HOME+'/data/temp/'
data = np.genfromtxt(dir_file+config_file, dtype=str)
config_data = {}
for ii in range(0,len(data[:,0])):
	str_temp = data[:,0][ii]
	config_data[str_temp] = data[:,1][ii]

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

global imf
imf = int(config_data['imf_type'])

# dust emission
global duste_switch
if int(config_data['duste_switch']) == 0:
	duste_switch = 'noduste'
elif int(config_data['duste_switch']) == 1:
	duste_switch = 'duste'

# dust extinction law
global dust_ext_law
if int(config_data['dust_ext_law']) == 0:
	dust_ext_law = 'CF2000'
elif int(config_data['dust_ext_law']) == 1:
	dust_ext_law = 'Cal2000'

# whether dust_index is set fix or not
global fix_dust_index, fix_dust_index_val
if float(config_data['pr_dust_index_min']) == float(config_data['pr_dust_index_max']):     # dust_index is fixed
	fix_dust_index = 1
	fix_dust_index_val = float(config_data['pr_dust_index_min'])
elif float(config_data['pr_dust_index_min']) != float(config_data['pr_dust_index_max']):   # dust_index varies
	fix_dust_index = 0
	fix_dust_index_val = 0

# AGN dusty torus
global add_agn 
add_agn = int(config_data['add_agn'])

# number of model SEDs that will be generated
global nmodels
nmodels = int(int(config_data['nmodels'])/size)*size

# name of output fits file
name_out = config_data['name_out']

# default parameter set
global def_params, def_params_val
def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma',
				'log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn']     # no normalization

def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

if fix_dust_index == 1:
	def_params_val['dust_index'] = fix_dust_index_val
if fix_dust_index == 0:
	def_params_val['dust_index'] = -0.7

global def_params_fsps, params_assoc_fsps, status_log
def_params_fsps = ['logzsol', 'log_tau', 'log_age', 'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 
				'log_qpah','log_fagn', 'log_tauagn']
params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

# call FSPS:
global sp 
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)
if duste_switch == 'duste':
	sp.params["add_dust_emission"] = True
elif duste_switch == 'noduste':
	sp.params["add_dust_emission"] = False
if add_neb_emission == 1:
	sp.params["add_neb_emission"] = True
	sp.params['gas_logu'] = gas_logu
elif add_neb_emission == 0:
	sp.params["add_neb_emission"] = False
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
free_z = 0
params0, nparams0 = get_params(free_z, sfh_form, duste_switch, dust_ext_law, add_agn, fix_dust_index)
params = params0[:int(nparams0-1)]   # exclude log_mass because models are normalized to solar mass
nparams = len(params)
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

global priors_min, priors_max
priors_min = np.zeros(nparams)
priors_max = np.zeros(nparams)
for ii in range(0,nparams): 
	str_temp = 'pr_%s_min' % params[ii]
	priors_min[ii] = float(config_data[str_temp])
	str_temp = 'pr_%s_max' % params[ii]
	priors_max[ii] = float(config_data[str_temp])

# Generate random models uniformly drawn within the input ranges
numDataPerRank = int(nmodels/size)
idx_mpi = np.linspace(0,nmodels-1,nmodels)
recvbuf_idx = np.empty(numDataPerRank, dtype='d')
comm.Scatter(idx_mpi, recvbuf_idx, root=0)

# get number of wavelength grids
global nwaves
params_val = def_params_val
for pp in range(0,nparams-1):  # log_mass is excluded
	params_val[params[pp]] = priors_min[pp]
wave, spec_flux, formed_mass, SFR_fSM, dust_mass, log_fagn_bol, mw_age = generate_modelSED_spec_restframe_props(sp=sp,imf_type=imf,sfh_form=sfh_form,
																				  add_agn=add_agn,params_fsps=params_fsps, params_val=params_val)
nwaves = len(wave)

mod_params_temp = np.zeros((nparams,numDataPerRank))
mod_log_mass_temp = np.zeros(numDataPerRank)
mod_log_sfr_temp = np.zeros(numDataPerRank)
mod_log_mw_age_temp = np.zeros(numDataPerRank)
if duste_switch == 'duste':
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
		temp_val = np.random.uniform(priors_min[pp],priors_max[pp],1)
		mod_params_temp[pp][int(count)] = temp_val
		params_val[params[pp]] = temp_val

	wave, spec_flux, formed_mass, SFR_fSM, dust_mass, log_fagn_bol, mw_age = generate_modelSED_spec_restframe_props(sp=sp,imf_type=imf,sfh_form=sfh_form,
																				  add_agn=add_agn,params_fsps=params_fsps, params_val=params_val)
	mod_log_mass_temp[int(count)] = log10(formed_mass)
	mod_log_sfr_temp[int(count)] = log10(SFR_fSM)
	mod_log_mw_age_temp[int(count)] = np.log10(mw_age)

	if duste_switch == 'duste':
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
if duste_switch == 'duste':
	mod_log_dustmass = np.zeros(nmodels)
if add_agn == 1:
	mod_log_fagn_bol = np.zeros(nmodels)

mod_params = np.zeros((nparams,nmodels))
mod_fluxes = np.zeros((nwaves,nmodels))

comm.Gather(mod_log_mass_temp, mod_log_mass, root=0)
comm.Gather(mod_log_sfr_temp, mod_log_sfr, root=0)
comm.Gather(mod_log_mw_age_temp, mod_log_mw_age, root=0)
if duste_switch == 'duste':
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
		m.attrs['dust_ext_law'] = dust_ext_law
		m.attrs['duste_switch'] = duste_switch
		m.attrs['add_neb_emission'] = add_neb_emission
		m.attrs['gas_logu'] = gas_logu
		m.attrs['add_agn'] = add_agn
		m.attrs['funit'] = 'L_sun/A'
		if duste_switch=='duste' or duste_switch==1:
			if fix_dust_index == 1:
				m.attrs['dust_index'] = fix_dust_index_val
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

		if duste_switch == 'duste':
			add_par = add_par + 1
			str_temp = 'par%d' % (nparams+add_par)
			m.attrs[str_temp] = 'log_dustmass'

		if add_agn == 1:
			add_par = add_par + 1
			str_temp = 'par%d' % (nparams+add_par)
			m.attrs[str_temp] = 'log_fagn_bol'

		m.attrs['nparams_all'] = int(nparams+add_par)

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

		if duste_switch == 'duste':
			p.create_dataset('log_dustmass', data=np.array(mod_log_dustmass), compression="gzip")

		if add_agn == 1:
			p.create_dataset('log_fagn_bol', data=np.array(mod_log_fagn_bol), compression="gzip") 

		# spectra
		s = m.create_group('spec')

		s.create_dataset('wave', data=np.array(mod_spec_wave), compression="gzip") 
		for ii in range(0,nmodels):
			str_temp = 'f%d' % ii
			s.create_dataset(str_temp, data=np.array(mod_fluxes_trans[ii]), compression="gzip")



