import numpy as np 
from math import log10, pow
import sys, os 
import fsps
from mpi4py import MPI
from astropy.io import fits
from astropy.cosmology import *

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.filtering import match_filters_array
from piXedfit.piXedfit_model import generate_modelSED_spec_fit, generate_modelSED_propphoto_nomwage_fit, calc_mw_age 
from piXedfit.piXedfit_fitting import get_params

## USAGE: mpirun -np [npros] python ./save_models_calc.py (1)name_filters_list (2)name_config

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# configuration file
global config_data
config_file = str(sys.argv[2])
dir_file = PIXEDFIT_HOME+'/data/temp/'
data = np.genfromtxt(dir_file+config_file, dtype=str)
config_data = {}
for ii in range(0,len(data[:,0])):
	str_temp = data[:,0][ii]
	config_data[str_temp] = data[:,1][ii]

# filters
global filters, nbands
dir_file = PIXEDFIT_HOME+'/data/temp/'
name_filters = str(sys.argv[1])
filters = np.genfromtxt(dir_file+name_filters, dtype=str)
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

# galaxy's redshift:
global gal_z
gal_z = float(config_data['gal_z'])

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

# igm absorption
global add_igm_absorption,igm_type
add_igm_absorption = int(config_data['add_igm_absorption'])
igm_type = int(config_data['igm_type'])

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
global npmod_seds
npmod_seds0 = int(config_data['npmod_seds'])

int_div = int(npmod_seds0/size)
npmod_seds = int_div*size
if rank == 0:
	print ("Number of random model SEDs to be generated: %d" % npmod_seds)

# name of output fits file
name_fits = config_data['name_out_fits']

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

global free_z, DL_Gpc, trans_fltr_int
free_z = 0
# calculate luminosity distance
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
def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma',
				'log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn']     # no normalization

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
params0, nparams0 = get_params(free_z, sfh_form, duste_switch, dust_ext_law, add_agn, fix_dust_index)
params = params0[:int(nparams0-1)]
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

# if redshift is fix: free_z=0, match wavelength points of transmission curves with that of spectrum at a given redshift
global trans_fltr_int
params_val = def_params_val
for pp in range(0,nparams-1):  # log_mass is excluded
	params_val[params[pp]] = priors_min[pp]

spec_SED = generate_modelSED_spec_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,add_igm_absorption=add_igm_absorption,
								igm_type=igm_type,params_fsps=params_fsps, params_val=params_val,DL_Gpc=DL_Gpc,
								cosmo=cosmo_str,H0=H0,Om0=Om0)

nwave = len(spec_SED['wave'])
if rank == 0:
	trans_fltr_int = match_filters_array(spec_SED['wave'],filters)
elif rank>0:
	trans_fltr_int = np.zeros((nbands,nwave))

comm.Bcast(trans_fltr_int, root=0)

# Generate random models uniformly drawn within the input ranges
numDataPerRank = int(npmod_seds/size)
idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
recvbuf_idx = np.empty(numDataPerRank, dtype='d')
comm.Scatter(idx_mpi, recvbuf_idx, root=0)

mod_params_temp = np.zeros((nparams,numDataPerRank))
mod_log_mass_temp = np.zeros(numDataPerRank)
mod_log_sfr_temp = np.zeros(numDataPerRank)
mod_log_mw_age_temp = np.zeros(numDataPerRank)
if duste_switch == 'duste':
	mod_log_dustmass_temp = np.zeros(numDataPerRank)
if add_agn == 1:
	mod_log_fagn_bol_temp = np.zeros(numDataPerRank)

mod_fluxes_temp = np.zeros((nbands,numDataPerRank))

count = 0
for ii in recvbuf_idx:
	params_val = def_params_val
	for pp in range(0,nparams): 
		temp_val = np.random.uniform(priors_min[pp],priors_max[pp],1)
		mod_params_temp[pp][int(count)] = temp_val
		params_val[params[pp]] = temp_val

	SED_prop,photo_SED = generate_modelSED_propphoto_nomwage_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,filters=filters,
												add_igm_absorption=add_igm_absorption,igm_type=igm_type,params_fsps=params_fsps,
												params_val=params_val,DL_Gpc=DL_Gpc,cosmo=cosmo_str,H0=H0,Om0=Om0,free_z=free_z,
												trans_fltr_int=trans_fltr_int)

	fluxes = photo_SED['flux']

	if np.isnan(SED_prop['SM'])==True or SED_prop['SM']<=0.0:
		mod_log_mass_temp[int(count)] = 1.0e-33
	else:
		mod_log_mass_temp[int(count)] = log10(SED_prop['SM'])

	if np.isnan(SED_prop['SFR'])==True or SED_prop['SFR']<=0.0:
		mod_log_sfr_temp[int(count)] = 1.0e-33
	else:
		mod_log_sfr_temp[int(count)] = log10(SED_prop['SFR'])

	if duste_switch == 'duste':
		if np.isnan(SED_prop['dust_mass'])==True or SED_prop['dust_mass']<=0.0:
			mod_log_dustmass_temp[int(count)] = 1.0e-33
		else:
			mod_log_dustmass_temp[int(count)] = log10(SED_prop['dust_mass'])  

	if add_agn == 1:
		mod_log_fagn_bol_temp[int(count)] = SED_prop['log_fagn_bol']

	# calculate mass-weighted age
	age = pow(10.0,params_val['log_age'])
	tau = pow(10.0,params_val['log_tau'])
	t0 = pow(10.0,params_val['log_t0'])
	alpha = pow(10.0,params_val['log_alpha'])
	beta = pow(10.0,params_val['log_beta'])
	mw_age = calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,age=age,formed_mass=SED_prop['SM'])
	mod_log_mw_age_temp[int(count)] = np.log10(mw_age)

	for bb in range(0,nbands):
		mod_fluxes_temp[bb][int(count)] = fluxes[bb]

	count = count + 1

	sys.stdout.write('\r')
	sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
	sys.stdout.flush()
sys.stdout.write('\n')

# gather the scattered data
mod_log_mass = np.zeros(npmod_seds)
mod_log_sfr = np.zeros(npmod_seds)
mod_log_mw_age = np.zeros(npmod_seds)
if duste_switch == 'duste':
	mod_log_dustmass = np.zeros(npmod_seds)
if add_agn == 1:
	mod_log_fagn_bol = np.zeros(npmod_seds)

mod_params = np.zeros((nparams,npmod_seds))
mod_fluxes = np.zeros((nbands,npmod_seds))

comm.Gather(mod_log_mass_temp, mod_log_mass, root=0)
comm.Gather(mod_log_sfr_temp, mod_log_sfr, root=0)
comm.Gather(mod_log_mw_age_temp, mod_log_mw_age, root=0)
if duste_switch == 'duste':
	comm.Gather(mod_log_dustmass_temp, mod_log_dustmass, root=0)
if add_agn == 1:
	comm.Gather(mod_log_fagn_bol_temp, mod_log_fagn_bol, root=0)

for pp in range(0,nparams):
	comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

for bb in range(0,nbands):
	comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)

# store into a FITS file
if rank == 0:
	hdr = fits.Header()
	hdr['imf_type'] = imf
	hdr['sfh_form'] = sfh_form
	hdr['dust_ext_law'] = dust_ext_law
	hdr['duste_switch'] = duste_switch
	hdr['add_neb_emission'] = add_neb_emission
	hdr['gas_logu'] = gas_logu
	hdr['add_agn'] = add_agn
	hdr['add_igm_absorption'] = add_igm_absorption
	if duste_switch == 'duste':
		if fix_dust_index == 1:
			hdr['dust_index'] = fix_dust_index_val
	hdr['cosmo'] = cosmo_str
	hdr['H0'] = H0
	hdr['Om0'] = Om0
	if add_igm_absorption == 1:
		hdr['igm_type'] = igm_type
	hdr['nfilters'] = nbands
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		hdr[str_temp] = filters[bb]
	hdr['gal_z'] = gal_z
	hdr['nrows'] = npmod_seds
	hdr['nparams'] = nparams
	for pp in range(0,nparams):
		str_temp = 'param%d' % pp
		hdr[str_temp] = params[pp]

	for pp in range(0,nparams):
		str_temp = 'pr_%s_min' % params[pp]
		hdr[str_temp] = priors_min[pp]
		str_temp = 'pr_%s_max' % params[pp]
		hdr[str_temp] = priors_max[pp]

	hdr['col1'] = 'id'
	col_count = 1
	for pp in range(0,nparams):
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = params[pp]

	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_mass'

	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_sfr'

	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'log_mw_age'

	if duste_switch == 'duste':
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = 'log_dustmass'

	if add_agn == 1:
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = 'log_fagn_bol'

	for bb in range(0,nbands):
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = filters[bb]

	hdr['ncols'] = col_count

	cols0 = []
	col = fits.Column(name='id', format='K', array=np.array(idx_mpi))
	cols0.append(col)

	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array(mod_params[pp]))
		cols0.append(col)

	col = fits.Column(name='log_mass', format='D', array=np.array(mod_log_mass))
	cols0.append(col)

	col = fits.Column(name='log_sfr', format='D', array=np.array(mod_log_sfr))
	cols0.append(col)

	col = fits.Column(name='log_mw_age', format='D', array=np.array(mod_log_mw_age))
	cols0.append(col)

	if duste_switch == 'duste':
		col = fits.Column(name='log_dustmass', format='D', array=np.array(mod_log_dustmass))
		cols0.append(col)

	if add_agn == 1:
		col = fits.Column(name='log_fagn_bol', format='D', array=np.array(mod_log_fagn_bol))
		cols0.append(col)

	for bb in range(0,nbands):
		col = fits.Column(name=filters[bb], format='D', array=np.array(mod_fluxes[bb]))
		cols0.append(col)

	cols = fits.ColDefs(cols0)
	hdu = fits.BinTableHDU.from_columns(cols)
	primary_hdu = fits.PrimaryHDU(header=hdr)

	hdul = fits.HDUList([primary_hdu, hdu])
	hdul.writeto(name_fits, overwrite=True)




