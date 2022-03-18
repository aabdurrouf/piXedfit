import numpy as np 
import math
import sys, os
import fsps
from mpi4py import MPI
from astropy.io import fits

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.piXedfit_model import csp_spec_restframe_fit, calc_bollum_from_spec_rest


## USAGE: mpirun -np [npros] python ./add_fagnbol.py (1)name_sampler_fits (2)fits_name_out

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta', 'log_age','dust_index','dust1','dust2',
			'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']
def_params_val={'log_mass':0.0,'z':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,'log_qpah':-99.0,'log_umin':-99.0,
				'log_gamma':-99.0,'dust1':-99.0,'dust2':-99.0, 'dust_index':-99.0,'log_age':-99.0,'log_alpha':-99.0,
				'log_beta':-99.0,'log_t0':-99.0,'log_tau':-99.0,'logzsol':-99.0}

def_params_fsps = ['logzsol', 'log_tau', 'log_age', 'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 
				'log_qpah','log_fagn', 'log_tauagn']
params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 'dust_index':"dust_index", 'dust1':"dust1", 
					'dust2':"dust2",'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 'log_qpah':"duste_qpah",
					'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,'log_gamma':1, 'log_umin':1, 
			'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

# get the fits file:
name_sampler_fits = str(sys.argv[1])
fits_name_out = str(sys.argv[2])

# open the FITS file
hdu = fits.open(name_sampler_fits)
header_samplers = hdu[0].header
data_samplers = hdu[1].data
hdu.close()

sfh_form = header_samplers['sfh_form']
dust_ext_law = header_samplers['dust_ext_law']
duste_switch = header_samplers['duste_stat'] 

add_neb_emission = int(header_samplers['add_neb_emission'])
gas_logu=-2.0

# parameters in the fitting
nparams = int(header_samplers['nparams'])
params = []
for pp in range(0,nparams):
	str_temp = 'param%d' % pp
	params.append(header_samplers[str_temp])
	
nsamplers = int(header_samplers['nrows'])
imf = int(header_samplers['imf'])

# AGN switch
add_agn = int(header_samplers['add_agn'])

# igm_absorption switch:
add_igm_absorption = int(header_samplers['add_igm_absorption'])
if add_igm_absorption == 1:
	igm_type = int(header_samplers['igm_type'])
elif add_igm_absorption == 0:
	igm_type = 0

# call fsps
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

params_fsps = []
for ii in range(0,len(def_params_fsps)):
	for jj in range(0,nparams):
		if def_params_fsps[ii] == params[jj]:
			params_fsps.append(params[jj])
nparams_fsps = len(params_fsps)

### divide into processes:
numDataPerRank = int(nsamplers/size)
min_pix_id_calc = 0
max_pix_id_calc = numDataPerRank*size - 1

calc_rest_stat = 1
if numDataPerRank*size == nsamplers:
	calc_rest_stat = 0

#idx_mpi = np.linspace(0,nsamplers-1,nsamplers)
idx_mpi = np.linspace(0,max_pix_id_calc,numDataPerRank*size)

recvbuf_idx = np.empty(numDataPerRank, dtype='d')
comm.Scatter(idx_mpi, recvbuf_idx, root=0)

mod_fagn_bol_temp = np.zeros(numDataPerRank)

count = 0
for ii in recvbuf_idx:

	# get stellar mass:
	formed_mass = math.pow(10.0,data_samplers['log_mass'][int(ii)])

	# input model parameters to FSPS:
	for pp in range(0,nparams_fsps):
		str_temp = params_assoc_fsps[params_fsps[pp]]
		if status_log[params_fsps[pp]] == 0:
			sp.params[str_temp] = data_samplers[params_fsps[pp]][int(ii)]
		elif status_log[params_fsps[pp]] == 1:
			sp.params[str_temp] = math.pow(10.0,data_samplers[params_fsps[pp]][int(ii)])

	sp.params['imf_type'] = imf
	# gas phase metallicity:
	sp.params['gas_logz'] = data_samplers['logzsol'][int(ii)]

	# get age:
	age = math.pow(10.0,data_samplers['log_age'][int(ii)])

	# get model rest-frame spectrum
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form==0 or sfh_form==1:
		
		wave, spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

		sp.params["fagn"] = 0.0
		wave, spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

	elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==2 or sfh_form==3 or sfh_form==4:
		if sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh':
			t0 = math.pow(10.0,data_samplers['log_t0'][int(ii)])
		elif sfh_form=='double_power_sfh':
			t0 = 0.0
		tau = math.pow(10.0,data_samplers['log_tau'][int(ii)])
		alpha = math.pow(10.0,data_samplers['log_alpha'][int(ii)])
		beta = math.pow(10.0,data_samplers['log_beta'][int(ii)])

		SFR_fSM,mass,wave,spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
		lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

		sp.params["fagn"] = 0.0
		SFR_fSM,mass,wave,spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
		lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

	
	mod_fagn_bol_temp[int(count)] = (lbol_agn-lbol_noagn)/lbol_agn

	count = count + 1
	sys.stdout.write('\r')
	sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
	sys.stdout.flush()
sys.stdout.write('\n')


# gather the scattered data
mod_fagn_bol = None
if rank == 0:
	mod_fagn_bol = np.empty(numDataPerRank*size, dtype='d')

comm.Gather(mod_fagn_bol_temp, mod_fagn_bol, root=0)

if rank == 0:

	if calc_rest_stat == 1:
		#===============================================================#
		# calculate the rest of the samplers
		mod_fagn_bol_temp = []

		count = 0
		for ii in range(max_pix_id_calc, nsamplers):
			# get stellar mass:
			formed_mass = math.pow(10.0,data_samplers['log_mass'][int(ii)])

			# input model parameters to FSPS:
			for pp in range(0,nparams_fsps):
				str_temp = params_assoc_fsps[params_fsps[pp]]
				if status_log[params_fsps[pp]] == 0:
					sp.params[str_temp] = data_samplers[params_fsps[pp]][int(ii)]
				elif status_log[params_fsps[pp]] == 1:
					sp.params[str_temp] = math.pow(10.0,data_samplers[params_fsps[pp]][int(ii)])

			sp.params['imf_type'] = imf
			# gas phase metallicity:
			sp.params['gas_logz'] = data_samplers['logzsol'][int(ii)]

			# get age:
			age = math.pow(10.0,data_samplers['log_age'][int(ii)])

			# get model rest-frame spectrum
			if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form==0 or sfh_form==1:
				
				wave, spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
				lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

				sp.params["fagn"] = 0.0
				wave, spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
				lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

			elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==2 or sfh_form==3 or sfh_form==4:
				if sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh':
					t0 = math.pow(10.0,data_samplers['log_t0'][int(ii)])
				elif sfh_form=='double_power_sfh':
					t0 = 0.0
				tau = math.pow(10.0,data_samplers['log_tau'][int(ii)])
				alpha = math.pow(10.0,data_samplers['log_alpha'][int(ii)])
				beta = math.pow(10.0,data_samplers['log_beta'][int(ii)])

				SFR_fSM,mass,wave,spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																					age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
				lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)


				sp.params["fagn"] = 0.0
				SFR_fSM,mass,wave,spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																					age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
				lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)


			mod_fagn_bol_temp.append((lbol_agn-lbol_noagn)/lbol_agn)

			count = count + 1
		#===============================================================#

		# merge together
		mod_fagn_bol_all = list(mod_fagn_bol) + mod_fagn_bol_temp

		mod_fagn_bol_all = np.asarray(mod_fagn_bol_all)

	else:
		mod_fagn_bol_all = mod_fagn_bol


	## Store to fits file:
	hdr = fits.Header()
	hdr['imf'] = imf
	hdr['nparams'] = header_samplers['nparams']
	hdr['sfh_form'] = header_samplers['sfh_form']
	hdr['dust_ext_law'] = header_samplers['dust_ext_law']
	hdr['nfilters'] = header_samplers['nfilters'] 
	hdr['duste_stat'] = header_samplers['duste_stat']
	hdr['add_neb_emission'] = header_samplers['add_neb_emission']
	hdr['add_agn'] = header_samplers['add_agn']
	hdr['add_igm_absorption'] = header_samplers['add_igm_absorption']

	if 'cosmo' in header_samplers:
		hdr['cosmo'] = header_samplers['cosmo']
		hdr['H0'] = header_samplers['H0']
		hdr['Om0'] = header_samplers['Om0']
	else:
		hdr['cosmo'] = 'flat_LCDM'
		hdr['H0'] = 70.0
		hdr['Om0'] = 0.3

	if 'dust_index' in header_samplers:
		hdr['dust_index'] = header_samplers['dust_index']
	if add_igm_absorption == 1:
		hdr['igm_type'] = header_samplers['igm_type']

	nbands = int(header_samplers['nfilters'])
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		hdr[str_temp] = header_samplers[str_temp]

		str_temp = 'flux%d' % bb
		hdr[str_temp] = header_samplers[str_temp]

		str_temp = 'flux_err%d' % bb 
		hdr[str_temp] = header_samplers[str_temp]

	hdr['free_z'] = header_samplers['free_z']
	if 'gal_z' in header_samplers:
		hdr['gal_z'] = header_samplers['gal_z']

	hdr['nrows'] = nsamplers
	# add parameters
	for pp in range(0,nparams):
		str_temp = 'param%d' % pp
		hdr[str_temp] = params[pp]

	col_count = 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'id'
	for pp in range(0,nparams):
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = params[pp]
	col_count = col_count + 1
	str_temp = 'col%d' % (col_count)
	hdr[str_temp] = 'log_sfr'
	col_count = col_count + 1
	str_temp = 'col%d' % (col_count)
	hdr[str_temp] = 'log_mw_age'

	if duste_switch == 'duste':
		col_count = col_count + 1
		str_temp = 'col%d' % (col_count)
		hdr[str_temp] = 'log_dustmass'

	if add_agn == 1:
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = 'log_fagn_bol'

	hdr['ncols'] = col_count

	cols0 = []

	col = fits.Column(name='id', format='K', array=np.array(data_samplers['id']))
	cols0.append(col)

	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array(data_samplers[params[pp]]))
		cols0.append(col)

	col = fits.Column(name='log_sfr', format='D', array=np.array(data_samplers['log_sfr']))
	cols0.append(col)

	col = fits.Column(name='log_mw_age', format='D', array=np.array(data_samplers['log_mw_age']))
	cols0.append(col)

	if duste_switch == 'duste':
		col = fits.Column(name='log_dustmass', format='D', array=np.array(data_samplers['log_dustmass']))
		cols0.append(col)

	if add_agn == 1:
		col = fits.Column(name='log_fagn_bol', format='D', array=np.array(np.log10(mod_fagn_bol_all)))
		cols0.append(col)

	cols = fits.ColDefs(cols0)

	hdu = fits.BinTableHDU.from_columns(cols)
	primary_hdu = fits.PrimaryHDU(header=hdr)

	hdul = fits.HDUList([primary_hdu, hdu])
	#fits_name_out = config_data['name_out_finalfit']
	hdul.writeto(fits_name_out, overwrite=True)




