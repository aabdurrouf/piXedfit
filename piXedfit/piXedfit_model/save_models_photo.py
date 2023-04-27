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

from piXedfit.utils.filtering import interp_filters_curves 
from piXedfit.piXedfit_model import * 

## USAGE: mpirun -np [npros] python ./save_models_photo.py (1)name_filters_list (2)name_config

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

config_file = str(sys.argv[2])
dir_file = PIXEDFIT_HOME+'/data/temp/'

imf, add_neb_emission, add_igm_absorption, igm_type, sfh_form, dust_law, duste_switch, add_agn, nmodels, free_gas_logz, smooth_velocity, sigma_smooth, smooth_lsf, name_file_lsf, name_out, params, nparams, priors_min, priors_max, cosmo_str, cosmo, H0, Om0, gal_z = read_config_file_genmodels(dir_file,config_file)

if rank == 0:
	print ("There are %d parameters: " % nparams)
	print (params)

# normalize with size
nmodels = int(nmodels/size)*size

params_fsps, nparams_fsps = get_params_fsps(params)
def_params_val = default_params_val()
def_params_val['z'] = gal_z

# check whether smoothing with a line spread function or not
if smooth_lsf == True or smooth_lsf == 1:
	data = np.loadtxt(dir_file+name_file_lsf)
	lsf_wave, lsf_sigma = data[:,0], data[:,1]
else:
	lsf_wave, lsf_sigma = None, None

# get list of filters
global filters, nbands
name_filters = str(sys.argv[1])
filters = np.genfromtxt(dir_file+name_filters, dtype=str)
nbands = len(filters)

global DL_Gpc, trans_fltr_int
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


# call FSPS:
global sp 
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)
sp = set_initial_fsps(sp,duste_switch,add_neb_emission,add_agn,sfh_form,dust_law,smooth_velocity=smooth_velocity,
					sigma_smooth=sigma_smooth,smooth_lsf=smooth_lsf,lsf_wave=lsf_wave,lsf_sigma=lsf_sigma)

# generate random parameters
rand_params = np.zeros((nparams,nmodels))
for pp in range(0,nparams):
	rand_params[pp] = np.random.uniform(priors_min[pp],priors_max[pp],nmodels)

global interp_filters_waves, interp_filters_trans
interp_filters_waves,interp_filters_trans = interp_filters_curves(filters)

# calculation
numDataPerRank = int(nmodels/size)
idx_mpi = np.linspace(0,nmodels-1,nmodels)
recvbuf_idx = np.empty(numDataPerRank, dtype='d')
comm.Scatter(idx_mpi, recvbuf_idx, root=0)

mod_params_temp = np.zeros((nparams,numDataPerRank))
mod_log_mass_temp = np.zeros(numDataPerRank)
mod_log_sfr_temp = np.zeros(numDataPerRank)
mod_log_mw_age_temp = np.zeros(numDataPerRank)
if duste_switch==1:
	mod_log_dustmass_temp = np.zeros(numDataPerRank)
if add_agn == 1:
	mod_log_fagn_bol_temp = np.zeros(numDataPerRank)

mod_fluxes_temp = np.zeros((nbands,numDataPerRank))

count = 0
for ii in recvbuf_idx:
	params_val = def_params_val
	for pp in range(0,nparams):
		temp_val = rand_params[pp][int(ii)]  
		mod_params_temp[pp][int(count)] = temp_val
		params_val[params[pp]] = temp_val

	SED_prop,fluxes = generate_modelSED_propphoto_nomwage_fit(sp=sp,imf_type=imf,sfh_form=sfh_form,filters=filters,
												add_igm_absorption=add_igm_absorption,igm_type=igm_type,params_fsps=params_fsps,
												params_val=params_val,DL_Gpc=DL_Gpc,cosmo=cosmo_str,H0=H0,Om0=Om0,
												interp_filters_waves=interp_filters_waves,
												interp_filters_trans=interp_filters_trans)

	if np.isnan(SED_prop['SM'])==True or SED_prop['SM']<=0.0:
		mod_log_mass_temp[int(count)] = 1.0e-33
	else:
		mod_log_mass_temp[int(count)] = log10(SED_prop['SM'])

	if np.isnan(SED_prop['SFR'])==True or SED_prop['SFR']<=0.0:
		mod_log_sfr_temp[int(count)] = 1.0e-33
	else:
		mod_log_sfr_temp[int(count)] = log10(SED_prop['SFR'])

	if duste_switch==1:
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
mod_log_mass = np.zeros(nmodels)
mod_log_sfr = np.zeros(nmodels)
mod_log_mw_age = np.zeros(nmodels)
if duste_switch==1:
	mod_log_dustmass = np.zeros(nmodels)
if add_agn == 1:
	mod_log_fagn_bol = np.zeros(nmodels)

mod_params = np.zeros((nparams,nmodels))
mod_fluxes = np.zeros((nbands,nmodels))

comm.Gather(mod_log_mass_temp, mod_log_mass, root=0)
comm.Gather(mod_log_sfr_temp, mod_log_sfr, root=0)
comm.Gather(mod_log_mw_age_temp, mod_log_mw_age, root=0)
if duste_switch==1:
	comm.Gather(mod_log_dustmass_temp, mod_log_dustmass, root=0)
if add_agn == 1:
	comm.Gather(mod_log_fagn_bol_temp, mod_log_fagn_bol, root=0)

for pp in range(0,nparams):
	comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)

for bb in range(0,nbands):
	comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)

# free_gas_logz, smooth_velocity, sigma_smooth, smooth_lsf, 
# store into a FITS file
if rank == 0:
	hdr = fits.Header()
	hdr['imf_type'] = imf
	hdr['sfh_form'] = sfh_form
	hdr['dust_law'] = dust_law
	hdr['duste_switch'] = duste_switch
	hdr['add_neb_emission'] = add_neb_emission
	hdr['add_agn'] = add_agn
	hdr['add_igm_absorption'] = add_igm_absorption
	hdr['free_gas_logz'] = free_gas_logz
	hdr['smooth_velocity'] =  smooth_velocity
	hdr['sigma_smooth'] = sigma_smooth
	hdr['smooth_lsf'] = smooth_lsf
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
	hdr['nrows'] = nmodels
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

	if duste_switch==1:
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

	if duste_switch==1:
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
	hdul.writeto(name_out, overwrite=True)




