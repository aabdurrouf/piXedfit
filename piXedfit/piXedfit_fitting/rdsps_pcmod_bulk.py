import numpy as np
from math import log10, pow 
import sys, os
import fsps
from operator import itemgetter
from mpi4py import MPI
from astropy.io import fits
from functools import reduce

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

from piXedfit.utils.posteriors import model_leastnorm, calc_chi2, gauss_prob, gauss_prob_reduced, student_t_prob
from piXedfit.piXedfit_model import calc_mw_age


def bayesian_sedfit_gauss():
	redcd_chi2 = float(config_data['redc_chi2_initfit'])

	numDataPerRank = int(npmod_seds/size)
	idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_fluxes_temp = np.zeros((nbands,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	sampler_log_mass_temp = np.zeros(numDataPerRank)
	sampler_log_sfr_temp = np.zeros(numDataPerRank)
	sampler_log_mw_age_temp = np.zeros(numDataPerRank)
	if duste_switch == 'duste':
		sampler_logdustmass_temp = np.zeros(numDataPerRank)
	if add_agn == 1:
		sampler_log_fagn_bol_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		for pp in range(0,nparams): 
			temp_val = data_randmod[params[pp]][int(ii)]
			mod_params_temp[pp][int(count)] = temp_val

		fluxes = np.zeros(nbands)
		for bb in range(0,nbands):
			fluxes[bb] = data_randmod[filters[bb]][int(ii)]

		norm0 = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		mod_fluxes0 = norm0*fluxes

		sampler_log_mass_temp[int(count)] = data_randmod['log_mass'][int(ii)]+log10(norm0)
		sampler_log_sfr_temp[int(count)] = data_randmod['log_sfr'][int(ii)]+log10(norm0)
		if duste_switch == 'duste':
			sampler_logdustmass_temp[int(count)] = data_randmod['log_dustmass'][int(ii)]+log10(norm0)
		if add_agn == 1:
			sampler_log_fagn_bol_temp[int(count)] = data_randmod['log_fagn_bol'][int(ii)]

		sampler_log_mw_age_temp[int(count)] = data_randmod['log_mw_age'][int(ii)]

		### calculate chi-square and prob.
		chi2 = calc_chi2(obs_fluxes,obs_flux_err,mod_fluxes0)

		if gauss_likelihood_form == 0:
			prob0 = gauss_prob(obs_fluxes,obs_flux_err,mod_fluxes0)
		elif gauss_likelihood_form == 1:
			prob0 = gauss_prob_reduced(obs_fluxes,obs_flux_err,mod_fluxes0)

		mod_chi2_temp[int(count)] = chi2
		mod_prob_temp[int(count)] = prob0
		for bb in range(0,nbands):
			mod_fluxes_temp[bb][int(count)] = mod_fluxes0[bb]

		count = count + 1
		
		#sys.stdout.write('\r')
		#sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		#sys.stdout.flush()
	#sys.stdout.write('\n')

	mod_params = np.zeros((nparams,numDataPerRank*size))
	mod_fluxes = np.zeros((nbands,numDataPerRank*size))
	mod_chi2 = np.zeros(numDataPerRank*size)
	mod_prob = np.zeros(numDataPerRank*size)

	sampler_log_mass = np.zeros(numDataPerRank*size)
	sampler_log_sfr = np.zeros(numDataPerRank*size)
	sampler_log_mw_age = np.zeros(numDataPerRank*size)
	if duste_switch == 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
	if add_agn == 1:
		sampler_log_fagn_bol = np.zeros(numDataPerRank*size)
				
	# gather the scattered data
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)
	for bb in range(0,nbands):
		comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)

	comm.Gather(sampler_log_mass_temp, sampler_log_mass, root=0)
	comm.Gather(sampler_log_sfr_temp, sampler_log_sfr, root=0)
	comm.Gather(sampler_log_mw_age_temp, sampler_log_mw_age, root=0)
	if duste_switch == 'duste':
		comm.Gather(sampler_logdustmass_temp, sampler_logdustmass, root=0)
	if add_agn == 1:
		comm.Gather(sampler_log_fagn_bol_temp, sampler_log_fagn_bol, root=0)
	
	status_add_err = np.zeros(1)
	modif_obs_flux_err = np.zeros(nbands)

	if rank == 0:
		idx0, min_val = min(enumerate(mod_chi2), key=itemgetter(1))

		fluxes = np.zeros(nbands)
		for bb in range(0,nbands):
			fluxes[bb] = mod_fluxes[bb][idx0]

		print ("reduced chi2 value of the best-fitting model: %lf" % (mod_chi2[idx0]/nbands))
		if mod_chi2[idx0]/nbands > redcd_chi2:   
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
				chi2 = calc_chi2(obs_fluxes,modif_obs_flux_err,fluxes)
				if chi2/nbands <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			print ("After adding %lf fraction to systematic error, reduced chi2 of best-fit model becomes: %lf" % (sys_err_frac,chi2/nbands))

			status_add_err[0] = 1

		elif mod_chi2[idx0]/nbands <= redcd_chi2:
			status_add_err[0] = 0

	# Broadcast
	comm.Bcast(status_add_err, root=0)

	if status_add_err[0] == 0:
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)
		if add_agn == 1:
			comm.Bcast(sampler_log_fagn_bol, root=0)

	elif status_add_err[0] == 1:
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(modif_obs_flux_err, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)
		if add_agn == 1:
			comm.Bcast(sampler_log_fagn_bol, root=0)

		mod_fluxes1 = np.transpose(mod_fluxes, axes=(1,0))     # become [idx-model][idx-band] 

		idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
		recvbuf_idx = np.empty(numDataPerRank, dtype='d')
		comm.Scatter(idx_mpi, recvbuf_idx, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		mod_prob_temp = np.zeros(numDataPerRank)

		count = 0
		for ii in recvbuf_idx:
			fluxes = mod_fluxes1[int(ii)]

			chi2 = calc_chi2(obs_fluxes,modif_obs_flux_err,fluxes)

			if gauss_likelihood_form == 0:
				prob0 = gauss_prob(obs_fluxes,modif_obs_flux_err,fluxes)
			elif gauss_likelihood_form == 1:
				prob0 = gauss_prob_reduced(obs_fluxes,modif_obs_flux_err,fluxes)

			mod_chi2_temp[int(count)] = chi2
			mod_prob_temp[int(count)] = prob0
			count = count + 1

		mod_prob = np.zeros(numDataPerRank*size)
		mod_chi2 = np.zeros(numDataPerRank*size)
				
		# gather the scattered data
		comm.Gather(mod_chi2_temp, mod_chi2, root=0)
		comm.Gather(mod_prob_temp, mod_prob, root=0)

		# Broadcast
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		# end of status_add_err[0] == 1

	if duste_switch != 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
	if add_agn != 1:
		sampler_log_fagn_bol = np.zeros(numDataPerRank*size)

	return mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass, sampler_log_fagn_bol


def bayesian_sedfit_student_t():
	redcd_chi2 = float(config_data['redc_chi2_initfit'])

	numDataPerRank = int(npmod_seds/size)
	idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_fluxes_temp = np.zeros((nbands,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	sampler_log_mass_temp = np.zeros(numDataPerRank)
	sampler_log_sfr_temp = np.zeros(numDataPerRank)
	sampler_log_mw_age_temp = np.zeros(numDataPerRank)
	if duste_switch == 'duste':
		sampler_logdustmass_temp = np.zeros(numDataPerRank)
	if add_agn == 1:
		sampler_log_fagn_bol_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		for pp in range(0,nparams): 
			temp_val = data_randmod[params[pp]][int(ii)]
			mod_params_temp[pp][int(count)] = temp_val

		fluxes = np.zeros(nbands)
		for bb in range(0,nbands):
			fluxes[bb] = data_randmod[filters[bb]][int(ii)]

		norm0 = model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
		mod_fluxes0 = norm0*fluxes

		sampler_log_mass_temp[int(count)] = data_randmod['log_mass'][int(ii)]+log10(norm0)
		sampler_log_sfr_temp[int(count)] = data_randmod['log_sfr'][int(ii)]+log10(norm0)
		if duste_switch == 'duste':
			sampler_logdustmass_temp[int(count)] = data_randmod['log_dustmass'][int(ii)]+log10(norm0)
		if add_agn == 1:
			sampler_log_fagn_bol_temp[int(count)] = data_randmod['log_fagn_bol'][int(ii)]

		sampler_log_mw_age_temp[int(count)] = data_randmod['log_mw_age'][int(ii)]

		# calculate chi-square and prob.
		chi2 = calc_chi2(obs_fluxes,obs_flux_err,mod_fluxes0)
		chi = (obs_fluxes-mod_fluxes0)/obs_flux_err
		prob0 = student_t_prob(dof,chi)

		mod_chi2_temp[int(count)] = chi2
		mod_prob_temp[int(count)] = prob0
		for bb in range(0,nbands):
			mod_fluxes_temp[bb][int(count)] = mod_fluxes0[bb]

		count = count + 1

		#sys.stdout.write('\r')
		#sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		#sys.stdout.flush()
	#sys.stdout.write('\n')

	mod_params = np.zeros((nparams,numDataPerRank*size))
	mod_fluxes = np.zeros((nbands,numDataPerRank*size))
	mod_chi2 = np.zeros(numDataPerRank*size)
	mod_prob = np.zeros(numDataPerRank*size)

	sampler_log_mass = np.zeros(numDataPerRank*size)
	sampler_log_sfr = np.zeros(numDataPerRank*size)
	sampler_log_mw_age = np.zeros(numDataPerRank*size)
	if duste_switch == 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
	if add_agn == 1:
		sampler_log_fagn_bol = np.zeros(numDataPerRank*size)

	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)
	for bb in range(0,nbands):
		comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)

	comm.Gather(sampler_log_mass_temp, sampler_log_mass, root=0)
	comm.Gather(sampler_log_sfr_temp, sampler_log_sfr, root=0)
	comm.Gather(sampler_log_mw_age_temp, sampler_log_mw_age, root=0)
	if duste_switch == 'duste':
		comm.Gather(sampler_logdustmass_temp, sampler_logdustmass, root=0)
	if add_agn == 1:
		comm.Gather(sampler_log_fagn_bol_temp, sampler_log_fagn_bol, root=0)

	status_add_err = np.zeros(1)
	modif_obs_flux_err = np.zeros(nbands)

	if rank == 0:
		idx0, min_val = min(enumerate(mod_chi2), key=itemgetter(1))

		fluxes = np.zeros(nbands)
		for bb in range(0,nbands):
			fluxes[bb] = mod_fluxes[bb][idx0]

		print ("reduced chi2 value of the best-fitting model: %lf" % (mod_chi2[idx0]/nbands))
		if mod_chi2[idx0]/nbands > redcd_chi2: 
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
				chi2 = calc_chi2(obs_fluxes,modif_obs_flux_err,fluxes)
				if chi2/nbands <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			print ("After adding %lf fraction to systematic error, reduced chi2 of best-fit model becomes: %lf" % (sys_err_frac,chi2/nbands))

			status_add_err[0] = 1

		elif mod_chi2[idx0]/nbands <= redcd_chi2:
			status_add_err[0] = 0

	comm.Bcast(status_add_err, root=0)

	if status_add_err[0] == 0:
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)
		if add_agn == 1:
			comm.Bcast(sampler_log_fagn_bol, root=0)

	elif status_add_err[0] == 1:
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(modif_obs_flux_err, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)
		if add_agn == 1:
			comm.Bcast(sampler_log_fagn_bol, root=0)

		# transpose
		mod_fluxes1 = np.transpose(mod_fluxes, axes=(1,0))     # become [idx-model][idx-band] 

		idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
		recvbuf_idx = np.empty(numDataPerRank, dtype='d')
		comm.Scatter(idx_mpi, recvbuf_idx, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		mod_prob_temp = np.zeros(numDataPerRank)

		count = 0
		for ii in recvbuf_idx:
			fluxes = mod_fluxes1[int(ii)]

			# calculate model's chi2 and prob.
			chi2 = calc_chi2(obs_fluxes,modif_obs_flux_err,fluxes)
			chi = (obs_fluxes-fluxes)/modif_obs_flux_err
			prob0 = student_t_prob(dof,chi)

			mod_chi2_temp[int(count)] = chi2
			mod_prob_temp[int(count)] = prob0
			count = count + 1

		mod_prob = np.zeros(numDataPerRank*size)
		mod_chi2 = np.zeros(numDataPerRank*size)

		comm.Gather(mod_chi2_temp, mod_chi2, root=0)
		comm.Gather(mod_prob_temp, mod_prob, root=0)

		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		# end of status_add_err[0] == 1

	if duste_switch != 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
	if add_agn != 1:
		sampler_log_fagn_bol = np.zeros(numDataPerRank*size)

	return mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass, sampler_log_fagn_bol

 
def store_to_fits(nsamples=None,sampler_params=None,sampler_log_mass=None,sampler_log_sfr=None,sampler_log_mw_age=None,
	sampler_logdustmass=None,sampler_log_fagn_bol=None,mod_chi2=None,mod_prob=None,cosmo_str='flat_LCDM',H0=70.0,Om0=0.3,
	name_sampler_fits0=None):

	sampler_id = np.linspace(1, nsamples, nsamples)

	hdr = fits.Header()
	hdr['imf'] = imf
	hdr['nparams'] = nparams
	hdr['sfh_form'] = sfh_form
	hdr['dust_ext_law'] = dust_ext_law
	hdr['nfilters'] = nbands
	hdr['duste_stat'] = duste_switch
	if duste_switch == 'duste':
		if fix_dust_index == 1:
			hdr['dust_index'] = fix_dust_index_val
	hdr['add_neb_emission'] = add_neb_emission
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
		str_temp = 'flux%d' % bb
		hdr[str_temp] = obs_fluxes[bb]
		str_temp = 'flux_err%d' % bb 
		hdr[str_temp] = obs_flux_err[bb]
	if free_z == 0:
		hdr['gal_z'] = gal_z
		hdr['free_z'] = 0
	elif free_z == 1:
		hdr['free_z'] = 1
	hdr['cosmo'] = cosmo_str
	hdr['H0'] = H0
	hdr['Om0'] = Om0
	hdr['nrows'] = nsamples
	hdr['nrows'] = nsamples
	# parameters
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

	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'chi2'
	col_count = col_count + 1
	str_temp = 'col%d' % col_count
	hdr[str_temp] = 'prob'

	hdr['ncols'] = col_count

	cols0 = []
	col = fits.Column(name='id', format='K', array=np.array(sampler_id))
	cols0.append(col)

	for pp in range(0,nparams):
		col = fits.Column(name=params[pp], format='D', array=np.array(sampler_params[params[pp]]))
		cols0.append(col)

	col = fits.Column(name='log_mass', format='D', array=np.array(sampler_log_mass))
	cols0.append(col)

	col = fits.Column(name='log_sfr', format='D', array=np.array(sampler_log_sfr))
	cols0.append(col)

	col = fits.Column(name='log_mw_age', format='D', array=np.array(sampler_log_mw_age))
	cols0.append(col)

	if duste_switch == 'duste':
		col = fits.Column(name='log_dustmass', format='D', array=np.array(sampler_logdustmass))
		cols0.append(col)

	if add_agn == 1:
		col = fits.Column(name='log_fagn_bol', format='D', array=np.array(sampler_log_fagn_bol))
		cols0.append(col)

	col = fits.Column(name='chi2', format='D', array=np.array(mod_chi2))
	cols0.append(col)

	col = fits.Column(name='prob', format='D', array=np.array(mod_prob))
	cols0.append(col)

	cols = fits.ColDefs(cols0)
	hdu = fits.BinTableHDU.from_columns(cols)
	primary_hdu = fits.PrimaryHDU(header=hdr)

	hdul = fits.HDUList([primary_hdu, hdu])
	hdul.writeto(name_sampler_fits0, overwrite=True)


"""
USAGE: mpirun -np [npros] python ./rdsps_pcmod_bulk.py (1)name_filters_list (2)name_config (3)name_inputSEDs (4)name_outs
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

# FITS file containing pre-calculated model SEDs
name_saved_randmod = config_data['name_saved_randmod']

# data of pre-calculated model SEDs
global header_randmod, data_randmod, npmod_seds
hdu = fits.open(name_saved_randmod)
header_randmod = hdu[0].header
data_randmod = hdu[1].data
hdu.close()

# number of model SEDs:
npmod_seds0 = int(header_randmod['nrows'])
npmod_seds = int(npmod_seds0/size)*size

global add_neb_emission
add_neb_emission = int(header_randmod['add_neb_emission'])

global gas_logu
gas_logu = float(header_randmod['gas_logu'])

global sfh_form
sfh_form = header_randmod['sfh_form']

# redshift
global gal_z
gal_z = float(header_randmod['gal_z'])

global imf
imf = int(header_randmod['imf_type'])

global duste_switch
duste_switch = header_randmod['duste_switch']

global fix_dust_index, fix_dust_index_val
if duste_switch == 'duste':
	if 'dust_index' in header_randmod:
		fix_dust_index_val = float(header_randmod['dust_index'])
		fix_dust_index = 1
	else:
		fix_dust_index_val = 0
		fix_dust_index = 0

global dust_ext_law
dust_ext_law = header_randmod['dust_ext_law']

global add_igm_absorption,igm_type
add_igm_absorption = int(header_randmod['add_igm_absorption'])
if add_igm_absorption == 1:
	igm_type = int(header_randmod['igm_type'])

global add_agn 
add_agn = int(header_randmod['add_agn'])

global likelihood_form
likelihood_form = config_data['likelihood']

# degree of freedom in the student's t likelihood function -> only relevant if likelihood_form='student_t'
global dof
dof = float(config_data['dof'])

global gauss_likelihood_form
gauss_likelihood_form = int(config_data['gauss_likelihood_form'])

# get input SEDs
data = np.genfromtxt(temp_dir+str(sys.argv[3]), dtype=str)
n_obs_sed = len(data[:,0])
bulk_obs_fluxes = np.zeros((n_obs_sed,nbands))
bulk_obs_flux_err = np.zeros((n_obs_sed,nbands))
for bb in range(0,nbands):
	rr = 0
	for ff in data[:,bb]:
		bulk_obs_fluxes[rr][bb] = float(ff)
		rr = rr + 1
	rr = 0
	for ff in data[:,bb+nbands]:
		bulk_obs_flux_err[rr][bb] = float(ff)
		rr = rr + 1

# names of output FITS files
name_outs = str(sys.argv[4])
name_sampler_fits = np.genfromtxt(temp_dir+name_outs, dtype=str)

global free_z
if gal_z <= 0:
	free_z = 1
else:
	free_z = 0

global cosmo_str, H0, Om0
cosmo_str = header_randmod['cosmo']
H0 = float(header_randmod['H0'])
Om0 = float(header_randmod['Om0'])

# list of parameters
global params, nparams
nparams = int(header_randmod['nparams'])
params = []
for ii in range(0,nparams):
	str_temp = 'param%d' % ii
	params.append(header_randmod[str_temp])

global obs_fluxes, obs_flux_err

# iteration for each SED
for ii in range(0,n_obs_sed):
	obs_fluxes = bulk_obs_fluxes[ii]
	obs_flux_err = bulk_obs_flux_err[ii]

	if likelihood_form == 'gauss':
		mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass, sampler_log_fagn_bol = bayesian_sedfit_gauss()
	elif likelihood_form == 'student_t':
		mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass, sampler_log_fagn_bol = bayesian_sedfit_student_t()
	else:
		print ("likelihood_form is not recognized!")
		sys.exit()

	if np.sum(modif_obs_flux_err) != 0:
		obs_flux_err = modif_obs_flux_err

	# change the format to dictionary
	nsamples = len(mod_params[0])
	sampler_params = {}
	for pp in range(0,nparams):
		sampler_params[params[pp]] = np.zeros(nsamples)
		sampler_params[params[pp]] = mod_params[pp]

	# store to FITS file
	if rank == 0:
		name_sampler_fits0 = name_sampler_fits[ii]
		store_to_fits(nsamples=nsamples,sampler_params=sampler_params,sampler_log_mass=sampler_log_mass,sampler_log_sfr=sampler_log_sfr,
					sampler_log_mw_age=sampler_log_mw_age,sampler_logdustmass=sampler_logdustmass, sampler_log_fagn_bol=sampler_log_fagn_bol,
					mod_chi2=mod_chi2,mod_prob=mod_prob, cosmo_str=cosmo_str,H0=H0,Om0=Om0,name_sampler_fits0=name_sampler_fits0)


