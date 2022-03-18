import numpy as np
import math
import sys
import fsps
from mpi4py import MPI
from astropy.io import fits
import operator
import filtering
import calc_posteriors
import redshifting
import piXedfit_model
import igm_absorption
import os
from astropy.cosmology import FlatLambdaCDM
from functools import reduce
from scipy import interpolate

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']

global cosmo
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

### define function for SED fitting with Bayesian statistic approach
### mod_fluxes = [idx-band][idx-model]  --> this input will only be used if increase_ferr==0
### mod_params = [idx-param][idx-model]  --> this input will only be used if increase_ferr==0
def bayesian_sedfit_gauss():
	## define reduced chi2 to be achieved:
	redcd_chi2 = 3.0
	## divide calculation to all the processes
	numDataPerRank = int(npmod_seds/size)

	idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
	## allocate memory in each process to receive the data:
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')     # allocate space for recvbuf
	## scatter the ids to the processes:
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_fluxes_temp = np.zeros((nbands,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	### some additional parameters:
	sampler_log_mass_temp = np.zeros(numDataPerRank)
	sampler_log_sfr_temp = np.zeros(numDataPerRank)
	sampler_log_mw_age_temp = np.zeros(numDataPerRank)
	if duste_switch == 'duste':
		sampler_logdustmass_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		### generate random redshifts:
		rand_z = np.random.uniform(priors_min_z,priors_max_z,nrand_z)
		### get parameters, except z. 
		### for each set of parameters, a best redshift will be choosen 
		for pp in range(0,nparams-1):    ### exclude redshift
			temp_val = saved_mod['z0'].data[params[pp]][int(ii)]
			mod_params_temp[pp][int(count)] = temp_val

		#### iterate for each random redshift:
		fluxes_rand_z = np.zeros((nrand_z,nbands))
		norm_rand_z = np.zeros(nrand_z)
		chi2_rand_z = np.zeros(nrand_z)
		for rr in range(0,nrand_z):
			### determine indexes of the grid_z that are going to be used for interpolation
			idx0 = np.interp(gal_z,rand_z[rr],idx_grid_z_temp)
			idx_grid_z0 = int(idx0)
			idx_grid_z1 = int(idx0) + 1

			idx_grid_z = []
			if idx_grid_z0 == 0:
				for aa in range(0,3):
					idx_grid_z.append(aa)
			elif idx_grid_z1 == ngrid_z:
				for aa in range(0,3):
					idx_grid_z.append(ngrid_z-aa)
			else:
				for aa in range(idx_grid_z0-1,idx_grid_z1+2):
					idx_grid_z.append()
			n_idx_grid_z = len(idx_grid_z)

			### make grid of redshifts for the interpolation:
			ref_grid_z = np.zeros(n_idx_grid_z)
			for aa in range(0,n_idx_grid_z):
				ref_grid_z[aa] = grid_z[int(idx_grid_z[aa])]

			### interpolation to get model fluxes at the galaxy's z:
			fluxes = np.zeros(nbands)
			for bb in range(0,nbands):
				ref_grid_fluxes = np.zeros(n_idx_grid_z)
				for zz in range(0,n_idx_grid_z):
					str_temp = 'z%d' % idx_grid_z[zz]
					ref_grid_fluxes[zz] = saved_mod[str_temp].data[filters[bb]][int(ii)]
				f = interpolate.CubicSpline(ref_grid_z,ref_grid_fluxes)
				fluxes[bb] = f(grid_z)
			### calculate model's best normalization:
			norm_rand_z[rr] = calc_posteriors.model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
			fluxes_rand_z[rr] = norm_rand_z[rr]*fluxes
			### calculate chi-square:
			chi2_rand_z[rr] = np.sum((fluxes_rand_z[rr]-obs_fluxes)*(fluxes_rand_z[rr]-obs_fluxes)/obs_flux_err/obs_flux_err)

		### select best redshift:
		idx_best, min_val = min(enumerate(chi2_rand_z), key=operator.itemgetter(1))
		### store the best redshift into the array:
		mod_params_temp[nparams-1][int(count)] = rand_z[idx_best]

		### calculate probbaility 
		norm0 = norm_rand_z[idx_best]
		mod_chi2_temp[int(count)] = chi2_rand_z[idx_best]
		mod_prob_temp[int(count)] = calc_posteriors.gauss_prob(obs_fluxes,obs_flux_err,fluxes_rand_z[idx_best])
		for bb in range(0,nbands):
			mod_fluxes_temp[bb][int(count)] = fluxes_rand_z[idx_best]

		### calculate other parameters:
		sampler_log_mass_temp[int(count)] = saved_mod['z0'].data['log_mass'][int(ii)]+math.log10(norm0)
		sampler_log_sfr_temp[int(count)] = saved_mod['z0'].data['log_sfr'][int(ii)]+math.log10(norm0)
		if duste_switch == 'duste':
			sampler_logdustmass_temp[int(count)] = saved_mod['z0'].data['log_dustmass'][int(ii)]+math.log10(norm0)

		### calculate MW-age:
		formed_mass = math.pow(10.0,saved_mod['z0'].data['log_mass'][int(ii)])
		age = math.pow(10.0,saved_mod['z0'].data['log_age'][int(ii)])
		tau = math.pow(10.0,saved_mod['z0'].data['log_tau'][int(ii)])
		t0 = 0.0
		alpha = 0.0
		beta = 0.0
		if sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
			t0 = math.pow(10.0,saved_mod['z0'].data['log_t0'][int(ii)])
		if sfh_form == 'double_power_sfh':
			alpha = math.pow(10.0,saved_mod['z0'].data['log_alpha'][int(ii)])
			beta = math.pow(10.0,saved_mod['z0'].data['log_beta'][int(ii)])
		mw_age = piXedfit_model.calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,
					age=age,formed_mass=formed_mass)
		sampler_log_mw_age_temp[int(count)] = math.log10(mw_age)
		
		count = count + 1

		#sys.stdout.write('\r')
		#sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		#sys.stdout.flush()
	#sys.stdout.write('\n')

	mod_params = np.zeros((nparams,numDataPerRank*size))
	mod_fluxes = np.zeros((nbands,numDataPerRank*size))
	mod_chi2 = np.zeros(numDataPerRank*size)
	mod_prob = np.zeros(numDataPerRank*size)

	### some additional parameters:
	sampler_log_mass = np.zeros(numDataPerRank*size)
	sampler_log_sfr = np.zeros(numDataPerRank*size)
	sampler_log_mw_age = np.zeros(numDataPerRank*size)
	if duste_switch == 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
				
	## gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)
	for bb in range(0,nbands):
		comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)
	### some additional properties:
	comm.Gather(sampler_log_mass_temp, sampler_log_mass, root=0)
	comm.Gather(sampler_log_sfr_temp, sampler_log_sfr, root=0)
	comm.Gather(sampler_log_mw_age_temp, sampler_log_mw_age, root=0)
	if duste_switch == 'duste':
		comm.Gather(sampler_logdustmass_temp, sampler_logdustmass, root=0)
	
	### allocate memory for possibly modified flux uncertainty
	status_add_err = np.zeros(1)
	modif_obs_flux_err = np.zeros(nbands)   ## default is set to 0

	if rank == 0:
		###=> get model with lowest chi-square: to modify flux uncertainties such that 
		### the best-fitting model should has reduced chi-square < redcd_chi2
		idx0, min_val = min(enumerate(mod_chi2), key=operator.itemgetter(1))
		print ("reduced chi2 value of the best-fitting model: %lf" % (mod_chi2[idx0]/nbands))
		if mod_chi2[idx0]/nbands > redcd_chi2:  
			print ("increasing iteratively flux error by adding a systematic error")
			mod_fluxes_trans = np.transpose(mod_fluxes, axes=(1,0))    ### [idx-mod][idx-band]
			mod_f0 = mod_fluxes_trans[idx0] 
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				# modifiy observed flux errors:
				modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
				chi2 = np.sum((mod_f0-obs_fluxes)*(mod_f0-obs_fluxes)/obs_flux_err/obs_flux_err)
				if chi2/nbands <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			print ("After adding %lf fraction to systematic error, reduced chi2 of best-fit model becomes: %lf" % (sys_err_frac,chi2/nbands))

			status_add_err[0] = 1

		elif mod_chi2[idx0]/nbands <= redcd_chi2:
			status_add_err[0] = 0

	## share the status to all processes:
	comm.Bcast(status_add_err, root=0)

	####### =========================== ########
	if status_add_err[0] == 0:
		## Broadcast to all processes:
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)
		 
	####### =========================== ########
	elif status_add_err[0] == 1:
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(modif_obs_flux_err, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)

		## transpose mod_fluxes0 if increase_ferr == 0:
		mod_fluxes1 = np.transpose(mod_fluxes, axes=(1,0))     ## become [idx-model][idx-band] 

		idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
		## allocate memory in each process to receive the data:
		recvbuf_idx = np.empty(numDataPerRank, dtype='d')       # allocate space for recvbuf
		## scatter the ids to the processes:
		comm.Scatter(idx_mpi, recvbuf_idx, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		mod_prob_temp = np.zeros(numDataPerRank)

		count = 0
		for ii in recvbuf_idx:
			## get model fluxes:
			fluxes = mod_fluxes1[int(ii)]
			## calculate model's probability:
			chi2 = np.sum((fluxes-obs_fluxes)*(fluxes-obs_fluxes)/modif_obs_flux_err/modif_obs_flux_err)
			#prob0 = gauss_prob(chi2)
			prob0 = calc_posteriors.gauss_prob(obs_fluxes,obs_flux_err,fluxes)
			mod_chi2_temp[int(count)] = chi2
			mod_prob_temp[int(count)] = prob0
			count = count + 1

		mod_prob = np.zeros(numDataPerRank*size)
		mod_chi2 = np.zeros(numDataPerRank*size)
				
		## gather the scattered data
		comm.Gather(mod_chi2_temp, mod_chi2, root=0)
		comm.Gather(mod_prob_temp, mod_prob, root=0)

		## Broadcast to all processes:
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		### end of status_add_err[0] == 1:
	####### =========================== ########
	if duste_switch != 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)

	return (mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass)

### define function for SED fitting with Bayesian statistic approach
### mod_fluxes = [idx-band][idx-model]  --> this input will only be used if increase_ferr==0
### mod_params = [idx-param][idx-model]  --> this input will only be used if increase_ferr==0
def bayesian_sedfit_student_t():
	## define reduced chi2 to be achieved:
	redcd_chi2 = 3.0
	## divide calculation to all the processes
	numDataPerRank = int(npmod_seds/size)

	idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
	## allocate memory in each process to receive the data:
	recvbuf_idx = np.empty(numDataPerRank, dtype='d')     # allocate space for recvbuf
	## scatter the ids to the processes:
	comm.Scatter(idx_mpi, recvbuf_idx, root=0)

	mod_params_temp = np.zeros((nparams,numDataPerRank))
	mod_fluxes_temp = np.zeros((nbands,numDataPerRank))
	mod_chi2_temp = np.zeros(numDataPerRank)
	mod_prob_temp = np.zeros(numDataPerRank)

	### some additional parameters:
	sampler_log_mass_temp = np.zeros(numDataPerRank)
	sampler_log_sfr_temp = np.zeros(numDataPerRank)
	sampler_log_mw_age_temp = np.zeros(numDataPerRank)
	if duste_switch == 'duste':
		sampler_logdustmass_temp = np.zeros(numDataPerRank)

	count = 0
	for ii in recvbuf_idx:
		### generate random redshifts:
		rand_z = np.random.uniform(priors_min_z,priors_max_z,nrand_z)
		### get parameters, except z. 
		### for each set of parameters, a best redshift will be choosen 
		for pp in range(0,nparams-1):    ### exclude redshift
			temp_val = saved_mod['z0'].data[params[pp]][int(ii)]
			mod_params_temp[pp][int(count)] = temp_val

		#### iterate for each random redshift:
		fluxes_rand_z = np.zeros((nrand_z,nbands))
		norm_rand_z = np.zeros(nrand_z)
		chi2_rand_z = np.zeros(nrand_z)
		for rr in range(0,nrand_z):
			### determine indexes of the grid_z that are going to be used for interpolation
			idx0 = np.interp(gal_z,rand_z[rr],idx_grid_z_temp)
			idx_grid_z0 = int(idx0)
			idx_grid_z1 = int(idx0) + 1

			idx_grid_z = []
			if idx_grid_z0 == 0:
				for aa in range(0,3):
					idx_grid_z.append(aa)
			elif idx_grid_z1 == ngrid_z:
				for aa in range(0,3):
					idx_grid_z.append(ngrid_z-aa)
			else:
				for aa in range(idx_grid_z0-1,idx_grid_z1+2):
					idx_grid_z.append()
			n_idx_grid_z = len(idx_grid_z)

			### make grid of redshifts for the interpolation:
			ref_grid_z = np.zeros(n_idx_grid_z)
			for aa in range(0,n_idx_grid_z):
				ref_grid_z[aa] = grid_z[int(idx_grid_z[aa])]

			### interpolation to get model fluxes at the galaxy's z:
			fluxes = np.zeros(nbands)
			for bb in range(0,nbands):
				ref_grid_fluxes = np.zeros(n_idx_grid_z)
				for zz in range(0,n_idx_grid_z):
					str_temp = 'z%d' % idx_grid_z[zz]
					ref_grid_fluxes[zz] = saved_mod[str_temp].data[filters[bb]][int(ii)]
				f = interpolate.CubicSpline(ref_grid_z,ref_grid_fluxes)
				fluxes[bb] = f(grid_z)
			### calculate model's best normalization:
			norm_rand_z[rr] = calc_posteriors.model_leastnorm(obs_fluxes,obs_flux_err,fluxes)
			fluxes_rand_z[rr] = norm_rand_z[rr]*fluxes
			### calculate chi-square:
			chi2_rand_z[rr] = np.sum((fluxes_rand_z[rr]-obs_fluxes)*(fluxes_rand_z[rr]-obs_fluxes)/obs_flux_err/obs_flux_err)

		### select best redshift:
		idx_best, min_val = min(enumerate(chi2_rand_z), key=operator.itemgetter(1))
		### store the best redshift into the array:
		mod_params_temp[nparams-1][int(count)] = rand_z[idx_best]

		### calculate probbaility 
		norm0 = norm_rand_z[idx_best]
		mod_chi2_temp[int(count)] = chi2_rand_z[idx_best]
		chi = (obs_fluxes-fluxes_rand_z[idx_best])/obs_flux_err
		prob0 = calc_posteriors.student_t_prob(dof,chi)
		mod_prob_temp[int(count)] = prob0
		for bb in range(0,nbands):
			mod_fluxes_temp[bb][int(count)] = fluxes_rand_z[idx_best]

		### calculate other parameters:
		sampler_log_mass_temp[int(count)] = saved_mod['z0'].data['log_mass'][int(ii)]+math.log10(norm0)
		sampler_log_sfr_temp[int(count)] = saved_mod['z0'].data['log_sfr'][int(ii)]+math.log10(norm0)
		if duste_switch == 'duste':
			sampler_logdustmass_temp[int(count)] = saved_mod['z0'].data['log_dustmass'][int(ii)]+math.log10(norm0)

		### calculate MW-age:
		formed_mass = math.pow(10.0,saved_mod['z0'].data['log_mass'][int(ii)])
		age = math.pow(10.0,saved_mod['z0'].data['log_age'][int(ii)])
		tau = math.pow(10.0,saved_mod['z0'].data['log_tau'][int(ii)])
		t0 = 0.0
		alpha = 0.0
		beta = 0.0
		if sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
			t0 = math.pow(10.0,saved_mod['z0'].data['log_t0'][int(ii)])
		if sfh_form == 'double_power_sfh':
			alpha = math.pow(10.0,saved_mod['z0'].data['log_alpha'][int(ii)])
			beta = math.pow(10.0,saved_mod['z0'].data['log_beta'][int(ii)])
		mw_age = piXedfit_model.calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,
					age=age,formed_mass=formed_mass)
		sampler_log_mw_age_temp[int(count)] = math.log10(mw_age)
		
		count = count + 1

		#sys.stdout.write('\r')
		#sys.stdout.write('rank: %d  Calculation process: %d from %d  --->  %d%%' % (rank,count,len(recvbuf_idx),count*100/len(recvbuf_idx)))
		#sys.stdout.flush()
	#sys.stdout.write('\n')

	mod_params = np.zeros((nparams,numDataPerRank*size))
	mod_fluxes = np.zeros((nbands,numDataPerRank*size))
	mod_chi2 = np.zeros(numDataPerRank*size)
	mod_prob = np.zeros(numDataPerRank*size)

	### some additional parameters:
	sampler_log_mass = np.zeros(numDataPerRank*size)
	sampler_log_sfr = np.zeros(numDataPerRank*size)
	sampler_log_mw_age = np.zeros(numDataPerRank*size)
	if duste_switch == 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)
				
	## gather the scattered data and collect to rank=0
	comm.Gather(mod_chi2_temp, mod_chi2, root=0)
	comm.Gather(mod_prob_temp, mod_prob, root=0)
	for pp in range(0,nparams):
		comm.Gather(mod_params_temp[pp], mod_params[pp], root=0)
	for bb in range(0,nbands):
		comm.Gather(mod_fluxes_temp[bb], mod_fluxes[bb], root=0)
	### some additional properties:
	comm.Gather(sampler_log_mass_temp, sampler_log_mass, root=0)
	comm.Gather(sampler_log_sfr_temp, sampler_log_sfr, root=0)
	comm.Gather(sampler_log_mw_age_temp, sampler_log_mw_age, root=0)
	if duste_switch == 'duste':
		comm.Gather(sampler_logdustmass_temp, sampler_logdustmass, root=0)
	
	### allocate memory for possibly modified flux uncertainty
	status_add_err = np.zeros(1)
	modif_obs_flux_err = np.zeros(nbands)   ## default is set to 0

	if rank == 0:
		###=> get model with lowest chi-square: to modify flux uncertainties such that 
		### the best-fitting model should has reduced chi-square < redcd_chi2
		idx0, min_val = min(enumerate(mod_chi2), key=operator.itemgetter(1))
		print ("reduced chi2 value of the best-fitting model: %lf" % (mod_chi2[idx0]/nbands))
		if mod_chi2[idx0]/nbands > redcd_chi2:  
			print ("increasing iteratively flux error by adding a systematic error")
			mod_fluxes_trans = np.transpose(mod_fluxes, axes=(1,0))    ### [idx-mod][idx-band]
			mod_f0 = mod_fluxes_trans[idx0] 
			sys_err_frac = 0.01
			while sys_err_frac <= 0.5:
				# modifiy observed flux errors:
				modif_obs_flux_err = np.sqrt(np.square(obs_flux_err) + np.square(sys_err_frac*obs_fluxes))
				chi2 = np.sum((mod_f0-obs_fluxes)*(mod_f0-obs_fluxes)/obs_flux_err/obs_flux_err)
				if chi2/nbands <= redcd_chi2:
					break
				sys_err_frac = sys_err_frac + 0.01
			print ("After adding %lf fraction to systematic error, reduced chi2 of best-fit model becomes: %lf" % (sys_err_frac,chi2/nbands))

			status_add_err[0] = 1

		elif mod_chi2[idx0]/nbands <= redcd_chi2:
			status_add_err[0] = 0

	## share the status to all processes:
	comm.Bcast(status_add_err, root=0)

	####### =========================== ########
	if status_add_err[0] == 0:
		## Broadcast to all processes:
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)
		 
	####### =========================== ########
	elif status_add_err[0] == 1:
		comm.Bcast(mod_params, root=0)
		comm.Bcast(mod_fluxes, root=0)
		comm.Bcast(modif_obs_flux_err, root=0)

		comm.Bcast(sampler_log_mass, root=0)
		comm.Bcast(sampler_log_sfr, root=0)
		comm.Bcast(sampler_log_mw_age, root=0)
		if duste_switch == 'duste':
			comm.Bcast(sampler_logdustmass, root=0)

		## transpose mod_fluxes0 if increase_ferr == 0:
		mod_fluxes1 = np.transpose(mod_fluxes, axes=(1,0))     ## become [idx-model][idx-band] 

		idx_mpi = np.linspace(0,npmod_seds-1,npmod_seds)
		## allocate memory in each process to receive the data:
		recvbuf_idx = np.empty(numDataPerRank, dtype='d')       # allocate space for recvbuf
		## scatter the ids to the processes:
		comm.Scatter(idx_mpi, recvbuf_idx, root=0)

		mod_chi2_temp = np.zeros(numDataPerRank)
		mod_prob_temp = np.zeros(numDataPerRank)

		count = 0
		for ii in recvbuf_idx:
			## get model fluxes:
			fluxes = mod_fluxes1[int(ii)]
			## calculate model's probability:
			chi2 = np.sum((fluxes-obs_fluxes)*(fluxes-obs_fluxes)/modif_obs_flux_err/modif_obs_flux_err)
			chi = (obs_fluxes-fluxes)/obs_flux_err
			prob0 = calc_posteriors.student_t_prob(dof,chi)
			mod_chi2_temp[int(count)] = chi2
			mod_prob_temp[int(count)] = prob0
			count = count + 1

		mod_prob = np.zeros(numDataPerRank*size)
		mod_chi2 = np.zeros(numDataPerRank*size)
				
		## gather the scattered data
		comm.Gather(mod_chi2_temp, mod_chi2, root=0)
		comm.Gather(mod_prob_temp, mod_prob, root=0)

		## Broadcast to all processes:
		comm.Bcast(mod_chi2, root=0)
		comm.Bcast(mod_prob, root=0)

		### end of status_add_err[0] == 1:
	####### =========================== ########
	if duste_switch != 'duste':
		sampler_logdustmass = np.zeros(numDataPerRank*size)

	return (mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass)


### define function to store the sampler chains into output fits file:
def store_to_fits(nsamples=None,sampler_params=None,sampler_log_mass=None,sampler_log_sfr=None,sampler_log_mw_age=None,
	sampler_logdustmass=None,mod_fluxes=None,mod_chi2=None,mod_prob=None): 
	### make sampler_id:
	sampler_id = np.linspace(1, nsamples, nsamples)

	hdr = fits.Header()
	#IMF: [0:Salpeter(1955), 1:Chabrier(2003), 2:Kroupa(2001)]
	hdr['imf'] = imf
	hdr['nparams'] = nparams
	hdr['sfh_form'] = sfh_form
	hdr['dust_ext_law'] = dust_ext_law
	hdr['nfilters'] = nbands
	hdr['duste_stat'] = duste_switch
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
	hdr['nrows'] = nsamples
	## add list of parameters:
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

	for bb in range(0,nbands):
		col_count = col_count + 1
		str_temp = 'col%d' % col_count
		hdr[str_temp] = filters[bb]

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

	for bb in range(0,nbands):
		col = fits.Column(name=filters[bb], format='D', array=np.array(mod_fluxes[bb]))
		cols0.append(col)

	col = fits.Column(name='chi2', format='D', array=np.array(mod_chi2))
	cols0.append(col)

	col = fits.Column(name='prob', format='D', array=np.array(mod_prob))
	cols0.append(col)


	cols = fits.ColDefs(cols0)
	hdu = fits.BinTableHDU.from_columns(cols)
	primary_hdu = fits.PrimaryHDU(header=hdr)

	hdul = fits.HDUList([primary_hdu, hdu])
	fits_name_out = config_data['name_out_finalfit']
	hdul.writeto(fits_name_out, overwrite=True)	


#######################################################################################################################
#################################################### MAIN PROGRAM #####################################################
#######################################################################################################################

"""
USAGE: mpirun -np [npros] python ./name_app.py (1)filters_list (2)configuration file
"""
if len(sys.argv)!=3:
	print ("## USAGE: mpirun -np [npros] python ./mcmcSEDfit.py (1)filters_list (2)configuration file")
	sys.exit()

global comm, size, rank
comm = MPI.COMM_WORLD
size = comm.Get_size() #gives number of ranks in comm
rank = comm.Get_rank() #give rank of the process

####### %%%%%%%%%%%%% GET INPUT %%%%%%%%%%%%%% #########
###==> get configuration file:
global config_data
config_file = str(sys.argv[2])
dir_file = PIXEDFIT_HOME+'/data/temp/'
data = np.genfromtxt(dir_file+config_file, dtype=str)
config_data = {}
for ii in range(0,len(data[:,0])):
	str_temp = data[:,0][ii]
	config_data[str_temp] = data[:,1][ii]
#get filters:
global filters, nbands
dir_file = PIXEDFIT_HOME+'/data/temp/'
name_filters = str(sys.argv[1])
filters = np.genfromtxt(dir_file+name_filters, dtype=str)
nbands = len(filters)
#flag for switching nebular emission:
global add_neb_emission
add_neb_emission = int(config_data['add_neb_emission'])
### get gas_logu:
global gas_logu
gas_logu = float(config_data['gas_logu'])

#get SFH form:
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
#get galaxy's redshift:
global gal_z
gal_z = float(config_data['gal_z'])
#get the input SED:
global obs_fluxes, obs_flux_err
obs_fluxes = np.zeros(nbands)
obs_flux_err = np.zeros(nbands)
for ii in range(0,nbands):
	str_temp = 'flux_%s' % filters[ii]
	obs_fluxes[ii] = float(config_data[str_temp])
	str_temp = 'flux_err_%s' % filters[ii]
	obs_flux_err[ii] = float(config_data[str_temp])

#get IMF: [0:Salpeter(1955), 1:Chabrier(2003), 2:Kroupa(2001)]
global imf
imf = int(config_data['imf_type'])
## get switch of dust emission modeling: ['duste': dust emission, 'noduste': no dust emission]
global duste_switch
if int(config_data['duste_switch']) == 0:
	duste_switch = 'noduste'
elif int(config_data['duste_switch']) == 1:
	duste_switch = 'duste'
## get dust extintion law: [0: Charlot&Fall(2000), 1: Calzetti+(2000)]
global dust_ext_law
if int(config_data['dust_ext_law']) == 0:
	dust_ext_law = 'CF2000'
elif int(config_data['dust_ext_law']) == 1:
	dust_ext_law = 'Cal2000'
## add igm absorption or not -> igm_type = [0:Madau(1995); 1:Inoue+(2014)]
global add_igm_absorption,igm_type
add_igm_absorption = int(config_data['add_igm_absorption'])
igm_type = int(config_data['igm_type'])

### determine whether dust_index is set fix or not:
global fix_dust_index, fix_dust_index_val
if float(config_data['pr_dust_index_min']) == float(config_data['pr_dust_index_max']):     ### dust_index is set fix
	fix_dust_index = 1
	fix_dust_index_val = float(config_data['pr_dust_index_min'])
elif float(config_data['pr_dust_index_min']) != float(config_data['pr_dust_index_max']):   ### dust_index is varies
	fix_dust_index = 0
	fix_dust_index_val = 0
### determine whether AGN modeling is turn on or off:
global add_agn 
add_agn = int(config_data['add_agn'])
### get likelihood form:
global likelihood_form
likelihood_form = config_data['likelihood']
### get degree of freedom of the student's t distribution --> only applicable if likelihood_form='student_t':
global dof
dof = float(config_data['dof'])
### get name of fits file containg pre-calculated random model SEDs:
name_saved_randmod = config_data['name_saved_randmod']
###### %%%%%%%%%%%%% End of GET INPUT %%%%%%%%%%%%%% #######

global free_z, DL_Gpc
if gal_z <= 0:
	free_z = 1
else:
	free_z = 0

## define number of parameters:
global params, nparams
if free_z == 0:
	if sfh_form == 'tau_sfh' or sfh_form == 'delayed_tau_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, mainSFH, noduste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_age','dust2']     ## nparams = 5
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, mainSFH, noduste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_age','dust2','log_fagn','log_tauagn']     ## nparams = 5
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_age','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, mainSFH, duste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_age','dust2','log_gamma','log_umin','log_qpah']
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, mainSFH, duste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)

	elif sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, otherSFH, noduste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_t0','log_age','dust2']
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, otherSFH, noduste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_fagn','log_tauagn']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, mainSFH, duste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_gamma','log_umin','log_qpah']
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, mainSFH, duste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)

	elif sfh_form == 'double_power_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, mainSFH, noduste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2']     ## nparams = 5
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, mainSFH, noduste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_fagn','log_tauagn']     ## nparams = 5
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, mainSFH, duste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_gamma','log_umin','log_qpah']
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, mainSFH, duste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn']
						nparams = len(params)

elif free_z == 1:
	if sfh_form == 'tau_sfh' or sfh_form == 'delayed_tau_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, mainSFH, noduste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_age','dust2','z']
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, mainSFH, noduste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_age','dust2','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_age','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, mainSFH, duste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_age','dust2','log_gamma','log_umin','log_qpah','z']
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, mainSFH, duste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
						nparams = len(params)

	elif sfh_form == 'log_normal_sfh' or sfh_form == 'gaussian_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, otherSFH, noduste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_t0','log_age','dust2','z']
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, otherSFH, noduste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, mainSFH, duste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_gamma','log_umin','log_qpah','z']
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, mainSFH, duste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_t0','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_t0','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']       ### 14 parameters
						nparams = len(params)

	elif sfh_form == 'double_power_sfh':
		if duste_switch == 'noduste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, mainSFH, noduste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','z']
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, mainSFH, noduste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, noduste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_fagn','log_tauagn','z']
						nparams = len(params)
		elif duste_switch == 'duste':
			if dust_ext_law == 'Cal2000':
				if add_agn == 0:
					## fix-z, mainSFH, duste, Cal2000, noAGN:
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_gamma','log_umin','log_qpah','z']
					nparams = len(params)
				elif add_agn == 1:
					## fix-z, mainSFH, duste, Cal2000, AGN:
					params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
					nparams = len(params)
			elif dust_ext_law == 'CF2000':
				if fix_dust_index == 1:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, noAGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, fix dust_index, AGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
						nparams = len(params)
				elif fix_dust_index == 0:
					if add_agn == 0:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, noAGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','z']
						nparams = len(params)
					elif add_agn == 1:
						## fix-z, mainSFH, duste, CF2000, vary dust_index, AGN:
						params = ['logzsol','log_tau','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma','log_umin','log_qpah','log_fagn','log_tauagn','z']
						nparams = len(params)

if rank == 0:
	print ("parameters set: ")
	print (params)
	print ("number of parameters: %d" % nparams)

###### define priors range for the parameters: including normalization
#global priors_min, priors_max
#priors_min = np.zeros(nparams)
#priors_max = np.zeros(nparams)
#or ii in range(0,nparams):      
#	str_temp = 'pr_%s_min' % params[ii]
#	priors_min[ii] = float(config_data[str_temp])
#	str_temp = 'pr_%s_max' % params[ii]
#	priors_max[ii] = float(config_data[str_temp])

#### get the prior range for redshift:
priors_min_z = float(config_data['pr_z_min'])
priors_max_z = float(config_data['pr_z_max'])

### get number of random redshift to be generated in each iteration:
nrand_z = int(config_data['nrand_z'])

#### =========== open the pre-calculated model SEDs ============ #####
global saved_mod, npmod_seds
saved_mod = fits.open(name_saved_randmod)
## get number of random model seds in each redshift grid:
npmod_seds0 = int(saved_mod[0].header['nrows'])
npmod_seds = int(npmod_seds0/size)*size
## get list of redshifts:
ngrid_z = int(saved_mod[0].header['nz'])
idx_grid_z_temp = np.zeros(ngrid_z)
grid_z = np.zeros(ngrid_z)
for ii in range(0,ngrid_z):
	str_temp = 'z%d' % ii
	grid_z[ii] = float(saved_mod[0].header[str_temp])
	idx_grid_z_temp[ii] = ii
#### =========== End of open the pre-calculated model SEDs ============ #####

if priors_min_z<min(grid_z) or priors_max_z>max(grid_z):
	print ("z_range must lie within the available redshift grids in %s ..." % name_saved_randmod)
	sys.exit()

#########=============== Perform bayesian SED fitting ===============############ 
if likelihood_form == 'gauss':
	mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass = bayesian_sedfit_gauss()
elif likelihood_form == 'student_t':
	mod_params, mod_fluxes, mod_chi2, mod_prob, modif_obs_flux_err, sampler_log_mass, sampler_log_sfr, sampler_log_mw_age, sampler_logdustmass = bayesian_sedfit_student_t()
else:
	print ("likelihood_form is not recognized!")
	sys.exit()

if np.sum(modif_obs_flux_err) != 0:
	obs_flux_err = modif_obs_flux_err
#########=============== End of Perform bayesian SED fitting ===============############

#########=============== calculate additional parameters ===============############
### change the format to dictionary:
nsamples = len(mod_params[0])
sampler_params = {}
for pp in range(0,nparams):
	sampler_params[params[pp]] = np.zeros(nsamples)
	sampler_params[params[pp]] = mod_params[pp]

### store to fits file:
if rank == 0:
	store_to_fits(nsamples=nsamples,sampler_params=sampler_params,sampler_log_mass=sampler_log_mass,sampler_log_sfr=sampler_log_sfr,
				sampler_log_mw_age=sampler_log_mw_age,sampler_logdustmass=sampler_logdustmass,mod_fluxes=mod_fluxes,mod_chi2=mod_chi2,
				mod_prob=mod_prob)


saved_mod.close()



