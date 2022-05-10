import numpy as np
import math
import sys, os
import random
import fsps
import operator
from astropy.io import fits
from scipy.interpolate import interp1d

from ..utils.redshifting import cosmo_redshifting
from ..utils.filtering import filtering, cwave_filters
from ..utils.igm_absorption import igm_att_madau, igm_att_inoue

# warning is not logged here. Perfect for clean unit test output
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0


__all__ = ["tau_SFH", "delay_tau_SFH", "lognormal_SFH", "gaussian_SFH", "double_power_SFH", "grid_SFH", "grid_arbitrary_SFH","calc_mw_age", "construct_SFH", 
			"construct_arbitrary_SFH", "calc_cumul_SM_t", "t_mass_formed", "frac_mass_stars", "csp_spec_restframe_fit", "get_dust_mass_othSFH_fit", 
			"get_sfr_dust_mass_fagnbol_othSFH_fit", "get_sfr_dust_mass_othSFH_fit", "get_dust_mass_mainSFH_fit", "get_dust_mass_fagnbol_mainSFH_fit", "spec_given_SFH_ZH", 
			"spec_given_ages_mass_Z", "convert_unit_spec_from_ergscm2A","get_nebem_wave", "get_no_nebem_wave", "get_no_nebem_wave_fit", "calc_bollum_from_spec", 
			"calc_bollum_from_spec_rest", "breakdown_CSP_into_SSP", "breakdown_CSP_into_SSP_restframe", "get_emlines_luminosities"]


def tau_SFH(tau,t):
	"""A function to calculate SFR at a time t in the tau-model SFH
	"""    
	np.seterr(all='ignore', divide='ignore', over='ignore')
	SFR = 1.0/np.exp(t/tau)    # in unit of M_sun/Gyr
	return SFR

def delay_tau_SFH(tau,t):
	"""A function to calculate SFR at a time t in the delayed-tau SFH model 
	"""
	np.seterr(all='ignore', divide='ignore', over='ignore')
	SFR = t/np.exp(t/tau)      # in unit of M_sun/Gyr
	return SFR

def lognormal_SFH(t0,tau,t):
	"""A function to calculate SFR at a time t in the log-normal SFH model
	"""
	np.seterr(all='ignore', divide='ignore', over='ignore')
	exp0 = (np.log(t)-t0)*(np.log(t)-t0)/2.0/tau/tau
	SFR = 1.0/t/np.exp(exp0)
	return SFR                 # in unit of M_sun/Gyr
 
def gaussian_SFH(t0,tau,t):
	"""A function to calculate SFR at a time t in the Gaussian SFH model
	"""
	np.seterr(all='ignore', divide='ignore', over='ignore')
	exp0 = (t-t0)*(t-t0)/2.0/tau/tau
	SFR = 1.0/np.exp(exp0)
	return SFR                 # in unit of M_sun/Gyr

def double_power_SFH(tau,alpha,beta,t):
	"""A function to calculate SFR at a time t in the Double power law SFH model 
	"""
	np.seterr(all='ignore', divide='ignore', over='ignore')
	c1 = np.power(t/tau,alpha)
	c2 = 1.0/np.power(t/tau,beta)
	SFR = 1.0/(c1 + c2)
	return SFR                 # in unit of M_sun/Gyr

def grid_SFH(sfh_form='delayed_tau_sfh',tau=0,t0=0,alpha=0,beta=0,age=0,formed_mass=0):
	np.seterr(all='ignore', divide='ignore', over='ignore')

	sfh_age = []
	sfh_sfr = []
	sfh_mass = []

	del_t = 0.001
	sfh_age = np.linspace(del_t,int(age/del_t)*del_t,int(age/del_t))

	t = age - sfh_age
	if sfh_form=='tau_sfh' or sfh_form==0:
		sfh_sfr = tau_SFH(tau,t)
	elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
		sfh_sfr = delay_tau_SFH(tau,t)
	elif sfh_form=='log_normal_sfh' or sfh_form==2:
		sfh_sfr = lognormal_SFH(t0,tau,t)
	elif sfh_form=='gaussian_sfh' or sfh_form==3:
		sfh_sfr = gaussian_SFH(t0,tau,t)
	elif sfh_form=='double_power_sfh' or sfh_form==4:
		sfh_sfr = double_power_SFH(tau,alpha,beta,t)
	else:
		print ("SFH choice is not recognized!")
		sys.exit()

	sfh_mass = sfh_sfr*del_t
	intg_mass = np.sum(sfh_mass)
	norm = formed_mass/intg_mass

	sfh_t = np.asarray(sfh_age)
	sfh_sfr = np.asarray(sfh_sfr)*norm/1e+9
	sfh_mass = np.asarray(sfh_mass)*norm
	return sfh_t, sfh_sfr, sfh_mass

 
def grid_arbitrary_SFH(age,sfh_t,sfh_sfr0):
	""" sfh_t goes forward from 0 to the age: linear
	"""
	del_t = 0.001
	sfh_age = np.linspace(del_t,int(age/del_t)*del_t,int(age/del_t))

	sfh_age0 = age - np.asarray(sfh_t)
	f = interp1d(sfh_age0, sfh_sfr0, fill_value="extrapolate")
	sfh_sfr = f(sfh_age)
	sfh_sfr[0] = sfh_sfr[1]
	sfh_sfr[len(sfh_age)-1] = sfh_sfr[len(sfh_age)-2]

	sfh_mass = sfh_sfr*del_t

	return sfh_age, sfh_sfr, sfh_mass


def calc_mw_age(sfh_form='delayed_tau_sfh',tau=0,t0=0,alpha=0,beta=0,age=0,formed_mass=0,sfh_t=[],sfh_sfr=[]):
	"""A function for calculating mass-weighted age of a given model SFH
	"""
	
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==0 or sfh_form==1 or sfh_form==2 or sfh_form==3 or sfh_form==4:
		sfh_age, sfh_sfr0, sfh_mass = grid_SFH(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,age=age,formed_mass=formed_mass)
	elif sfh_form=='arbitrary_sfh':
		sfh_age, sfh_sfr0, sfh_mass = grid_arbitrary_SFH(age,sfh_t,sfh_sfr)
	else:
		print ("SFH choice is not recognized!")
		sys.exit()

	mod_mw_age = np.sum(sfh_mass*sfh_age)/np.sum(sfh_mass)

	return mod_mw_age


def construct_SFH(sfh_form='log_normal_sfh',t0=0.0,tau=0.0,alpha=0.0,beta=0.0,age=0.0,formed_mass=1.0,del_t=0.001):
	"""A function to construct a parametric SFH model

	:param sfh_form:
		A choice of parametric SFH model. Options are: ['tau_sfh', 'delayed_tau_sfh', 'log_normal_sfh', 'gaussian_sfh', 'double_power_sfh']
	"""
	np.seterr(all='ignore', divide='ignore', over='ignore')

	t = np.linspace(del_t,age,int((age-del_t)/del_t)+1)

	if sfh_form=='tau_sfh' or sfh_form==0:
		SFR_t0 = tau_SFH(tau,t)
	elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
		SFR_t0 = delay_tau_SFH(tau,t)
	elif sfh_form=='log_normal_sfh' or sfh_form==2:
		SFR_t0 = lognormal_SFH(t0,tau,t)
	elif sfh_form=='gaussian_sfh' or sfh_form==3:
		SFR_t0 = gaussian_SFH(t0,tau,t)
	elif sfh_form=='double_power_sfh' or sfh_form==4:
		SFR_t0 = double_power_SFH(tau,alpha,beta,t)
	else:
		print ("SFH choice is not recognized!")
		sys.exit()

	SFR_t0_clean = SFR_t0[np.logical_not(np.isnan(SFR_t0))]
	intg_mass = np.sum(SFR_t0_clean*del_t)
	SFR_t = SFR_t0*formed_mass/intg_mass/1e+9 

	return t,SFR_t

def construct_arbitrary_SFH(age,sfh_t,sfh_sfr):
	"""A function to construct arbitrary (i.e., non-functional form) SFH model
	"""
	npoints = len(sfh_t)

	del_t = 0.001
	t = np.linspace(del_t,age,int((age-del_t)/del_t)+1)

	f = interp1d(sfh_t,sfh_sfr, fill_value="extrapolate")
	SFR_t = f(t)
	SFR_t[0] = SFR_t[1]
	SFR_t[len(t)-1] = SFR_t[len(t)-2]

	return t, SFR_t


def calc_cumul_SM_t(sfh_form='delayed_tau_sfh',tau=0,t0=0,alpha=0,beta=0,age=None,formed_mass=0,sfh_t=[],sfh_sfr=[],del_t=0.001):
	"""A function to calculate history of stellar mass growth
	"""
	if age == None:
		age = max(sfh_t)

	# make grid of SFH with time
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==0 or sfh_form==1 or sfh_form==2 or sfh_form==3 or sfh_form==4:
		t,SFR_t = construct_SFH(sfh_form=sfh_form,t0=t0,tau=tau,alpha=alpha,beta=beta,age=age,formed_mass=formed_mass)
	elif sfh_form=='arbitrary_sfh':
		t,SFR_t = construct_arbitrary_SFH(age=age,sfh_t=sfh_t,sfh_sfr=sfh_sfr)
	else:
		print ("SFH choice is not recognized!")
		sys.exit()

	ntimes = len(t)

	SM_t = SFR_t*del_t*1.0e+9

	cumul_SM_t = np.cumsum(SM_t)
	sSFR_t = SFR_t*1.0e+9/cumul_SM_t
	 
	return (t,SFR_t,SM_t,sSFR_t,cumul_SM_t)


def t_mass_formed(t_fwd=[],cumul_SM_t=[],age=None,perc=50.0):
	"""A function for calculating time at which a given fraction of current M* formed.
	"""
	if age == None:
		age = max(t_fwd)
	total_SM = max(cumul_SM_t)
	crit_SM = perc*total_SM/100.0

	del_SM = np.abs(crit_SM - cumul_SM_t)
	idx, min_val = min(enumerate(del_SM), key=operator.itemgetter(1))

	t_fwd_form = t_fwd[idx]
	lbt_form = age - t_fwd_form
	
	return t_fwd_form, lbt_form

def frac_mass_stars(SFH_SFR,SFH_t,age_max):
	"""A function for calculating a fraction of mass associated with stars with ages below a certain value:
	## SFH_SFR and SFH_t is going backward in time, such that SFH_t is array of look-back time and SFH_SFR[0] is current SFR:
	## SFH_t should be in Gyr and SFH_SFR should be in M0/yr
	"""
	ntimes = len(SFH_t)
	tot_mass = 0
	tot_mass_below_age = 0
	for ii in range(0,ntimes-1):
		grid_mass = 0.5*(SFH_SFR[ii]+SFH_SFR[ii+1])*(SFH_t[ii+1]-SFH_t[ii])*1.0e+9
		if SFH_t[ii+1]<=age_max:
			tot_mass_below_age = tot_mass_below_age + grid_mass
		tot_mass = tot_mass + grid_mass

	frac_mass = tot_mass_below_age/tot_mass 
	return frac_mass


def csp_spec_restframe_fit(sp=None,sfh_form='delayed_tau_sfh',formed_mass=1.0,age=0.0,
	tau=0.0,t0=0.0,alpha=0.0,beta=0.0):
	"""A function for generating model spectrum of an CSP

	:param sp:
		Initialization of FSPS, such as sp=fsps.StellarPopulation()

	:param sfh_form (default: 'delayed_tau_sfh'):
		Choice for the parametric SFH model. 
		Options are: ['tau_sfh', 'delayed_tau_sfh', 'log_normal_sfh', 'gaussian_sfh', 'double_power_sfh']

	:param formed_mass:
		The total stellar mass formed.
	"""

	sp.params["sfh"] = 3

	# make grid of SFH:
	sfh_t, sfh_sfr = construct_SFH(sfh_form=sfh_form,t0=t0,tau=tau,alpha=alpha,
										beta=beta,age=age,formed_mass=formed_mass)

	if np.isnan(sfh_sfr).any()==True or np.isinf(sfh_sfr).any()==True:
		idx_excld = np.where((np.isnan(sfh_sfr)==True) | (np.isinf(sfh_sfr)==True))
		sfh_sfr[idx_excld[0]] = 0.0

		if np.any(sfh_sfr > 1.0e-33) == False:
			if len(idx_excld[0]) <= len(sfh_sfr) - 3:
				scale_sfr = 1.0/max(sfh_sfr)
				sfh_sfr = sfh_sfr*scale_sfr

				sp.set_tabular_sfh(sfh_t, sfh_sfr)

				wave, spec0 = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
				mass0 = sp.stellar_mass
				dust_mass0 = sp.dust_mass

				norm = formed_mass/mass0
				spec = spec0*norm/scale_sfr
				dust_mass = dust_mass0*norm/scale_sfr

				#SFR = sp.sfr*formed_mass
				SFR = sp.sfr/scale_sfr
				mass = formed_mass
			else:
				SFR = 1.0e-33
				mass = formed_mass
				wave = np.zeros(5994)
				spec = np.zeros(5994)
				dust_mass = 1.0e-33
		else:
			sp.set_tabular_sfh(sfh_t, sfh_sfr)

			wave, spec0 = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass

			norm = formed_mass/mass0
			spec = spec0*norm
			dust_mass = dust_mass0*norm

			SFR = sp.sfr
			mass = formed_mass

	else:
		if np.any(sfh_sfr > 1.0e-33) == False:
			if len(sfh_sfr)<=3:
				SFR = 1.0e-33
				mass = formed_mass
				wave = np.zeros(5994)
				spec = np.zeros(5994)
				dust_mass = 1.0e-33
			else:
				scale_sfr = 1.0/max(sfh_sfr)
				sfh_sfr = sfh_sfr*scale_sfr

				sp.set_tabular_sfh(sfh_t, sfh_sfr)

				wave, spec0 = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
				mass0 = sp.stellar_mass
				dust_mass0 = sp.dust_mass

				norm = formed_mass/mass0
				spec = spec0*norm/scale_sfr
				dust_mass = dust_mass0*norm/scale_sfr

				#SFR = sp.sfr*formed_mass
				SFR = sp.sfr/scale_sfr
				mass = formed_mass

		else:
			sp.set_tabular_sfh(sfh_t, sfh_sfr)

			wave, spec0 = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass

			norm = formed_mass/mass0
			spec = spec0*norm
			dust_mass = dust_mass0*norm

			SFR = sp.sfr
			mass = formed_mass

	return SFR,mass,wave,spec,dust_mass

def get_dust_mass_othSFH_fit(sp=None,imf_type=1,sfh_form='log_normal_sfh',params_fsps=['logzsol', 'log_tau', 'log_age', 
	'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 'log_qpah','log_fagn', 'log_tauagn'], params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):
	
	params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
	status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

	# get stellar mass:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	t0 = math.pow(10.0,params_val['log_t0'])
	tau = math.pow(10.0,params_val['log_tau'])
	age = math.pow(10.0,params_val['log_age'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	# input model parameters to FSPS:
	nparams_fsps = len(params_fsps)
	for pp in range(0,nparams_fsps):
		str_temp = params_assoc_fsps[params_fsps[pp]]
		if status_log[params_fsps[pp]] == 0:
			sp.params[str_temp] = params_val[params_fsps[pp]]
		elif status_log[params_fsps[pp]] == 1:
			sp.params[str_temp] = math.pow(10.0,params_val[params_fsps[pp]])

	sp.params['imf_type'] = imf_type
	# gas phase metallicity:
	sp.params['gas_logz'] = params_val['logzsol']

	# CSP
	sp.params["sfh"] = 3

	# make grid of SFH:
	sfh_t, sfh_sfr = construct_SFH(sfh_form=sfh_form,t0=t0,tau=tau,alpha=alpha,
										beta=beta,age=age,formed_mass=formed_mass)

	if np.isnan(sfh_sfr).any()==True or np.isinf(sfh_sfr).any()==True:
		idx_excld = np.where((np.isnan(sfh_sfr)==True) | (np.isinf(sfh_sfr)==True))
		sfh_sfr[idx_excld[0]] = 0.0

		if np.any(sfh_sfr > 1.0e-33) == False:
			if len(idx_excld[0]) <= len(sfh_sfr) - 3:
				scale_sfr = 1.0/max(sfh_sfr)
				sfh_sfr = sfh_sfr*scale_sfr

				sp.set_tabular_sfh(sfh_t, sfh_sfr)
				mass0 = sp.stellar_mass
				dust_mass0 = sp.dust_mass
				norm = formed_mass/mass0
				dust_mass = dust_mass0*norm/scale_sfr
			else:
				dust_mass = 1.0e-33
		else:
			sp.set_tabular_sfh(sfh_t, sfh_sfr)
			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass
			norm = formed_mass/mass0
			dust_mass = dust_mass0*norm
	else:
		if np.any(sfh_sfr > 1.0e-33) == False:
			scale_sfr = 1.0/max(sfh_sfr)
			sfh_sfr = sfh_sfr*scale_sfr

			sp.set_tabular_sfh(sfh_t, sfh_sfr)
			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass
			norm = formed_mass/mass0
			dust_mass = dust_mass0*norm/scale_sfr
		else:
			sp.set_tabular_sfh(sfh_t, sfh_sfr)
			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass
			norm = formed_mass/mass0
			dust_mass = dust_mass0*norm

	return dust_mass


def get_sfr_dust_mass_fagnbol_othSFH_fit(sp=None,imf_type=1,sfh_form='log_normal_sfh',params_fsps=['logzsol', 'log_tau', 'log_age', 
	'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 'log_qpah','log_fagn', 'log_tauagn'],params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):
	
	params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
	status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

	# get stellar mass:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	t0 = math.pow(10.0,params_val['log_t0'])
	tau = math.pow(10.0,params_val['log_tau'])
	age = math.pow(10.0,params_val['log_age'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	# input model parameters to FSPS:
	nparams_fsps = len(params_fsps)
	for pp in range(0,nparams_fsps):
		str_temp = params_assoc_fsps[params_fsps[pp]]
		if status_log[params_fsps[pp]] == 0:
			sp.params[str_temp] = params_val[params_fsps[pp]]
		elif status_log[params_fsps[pp]] == 1:
			sp.params[str_temp] = math.pow(10.0,params_val[params_fsps[pp]])

	sp.params['imf_type'] = imf_type
	# gas phase metallicity:
	sp.params['gas_logz'] = params_val['logzsol']

	# CSP
	sp.params["sfh"] = 3

	# make grid of SFH:
	sfh_t, sfh_sfr = construct_SFH(sfh_form=sfh_form,t0=t0,tau=tau,alpha=alpha,
										beta=beta,age=age,formed_mass=formed_mass)

	if np.isnan(sfh_sfr).any()==True or np.isinf(sfh_sfr).any()==True:
		idx_excld = np.where((np.isnan(sfh_sfr)==True) | (np.isinf(sfh_sfr)==True))
		sfh_sfr[idx_excld[0]] = 0.0

		if np.any(sfh_sfr > 1.0e-33) == False:
			if len(idx_excld[0]) <= len(sfh_sfr) - 3:
				scale_sfr = 1.0/max(sfh_sfr)
				sfh_sfr = sfh_sfr*scale_sfr

				sp.set_tabular_sfh(sfh_t, sfh_sfr)

				mass0 = sp.stellar_mass
				dust_mass0 = sp.dust_mass
				norm = formed_mass/mass0
				dust_mass = dust_mass0*norm/scale_sfr
				SFR = sp.sfr/scale_sfr

				# calculate AGN lum.
				wave, spec = sp.get_spectrum(peraa=True,tage=age)
				lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)
				sp.params["fagn"] = 0.0
				wave, spec = sp.get_spectrum(peraa=True,tage=age)
				lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

				log_fagn_bol = np.log10((lbol_agn-lbol_noagn)/lbol_agn)

			else:
				SFR = 1.0e-33
				dust_mass = 1.0e-33
				log_fagn_bol = 1.0e-33
		else:
			sp.set_tabular_sfh(sfh_t, sfh_sfr)
			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass
			norm = formed_mass/mass0
			dust_mass = dust_mass0*norm
			SFR = sp.sfr

			# calculate AGN lum.
			wave, spec = sp.get_spectrum(peraa=True,tage=age)
			lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)
			sp.params["fagn"] = 0.0
			wave, spec = sp.get_spectrum(peraa=True,tage=age)
			lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

			log_fagn_bol = np.log10((lbol_agn-lbol_noagn)/lbol_agn)

	else:
		if np.any(sfh_sfr > 1.0e-33) == False:
			scale_sfr = 1.0/max(sfh_sfr)
			sfh_sfr = sfh_sfr*scale_sfr

			sp.set_tabular_sfh(sfh_t, sfh_sfr)
			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass
			norm = formed_mass/mass0
			dust_mass = dust_mass0*norm/scale_sfr
			SFR = sp.sfr/scale_sfr

			# calculate AGN lum.
			wave, spec = sp.get_spectrum(peraa=True,tage=age)
			lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)
			sp.params["fagn"] = 0.0
			wave, spec = sp.get_spectrum(peraa=True,tage=age)
			lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

			log_fagn_bol = np.log10((lbol_agn-lbol_noagn)/lbol_agn)

		else:
			sp.set_tabular_sfh(sfh_t, sfh_sfr)
			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass
			norm = formed_mass/mass0
			dust_mass = dust_mass0*norm
			SFR = sp.sfr

			# calculate AGN lum.
			wave, spec = sp.get_spectrum(peraa=True,tage=age)
			lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)
			sp.params["fagn"] = 0.0
			wave, spec = sp.get_spectrum(peraa=True,tage=age)
			lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

			log_fagn_bol = np.log10((lbol_agn-lbol_noagn)/lbol_agn)

	return SFR, dust_mass, log_fagn_bol


def get_sfr_dust_mass_othSFH_fit(sp=None,imf_type=1,sfh_form='log_normal_sfh',params_fsps=['logzsol', 'log_tau', 'log_age', 
	'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 'log_qpah','log_fagn', 'log_tauagn'], params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):
	
	params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
	status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

	# get stellar mass:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	t0 = math.pow(10.0,params_val['log_t0'])
	tau = math.pow(10.0,params_val['log_tau'])
	age = math.pow(10.0,params_val['log_age'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	# input model parameters to FSPS:
	nparams_fsps = len(params_fsps)
	for pp in range(0,nparams_fsps):
		str_temp = params_assoc_fsps[params_fsps[pp]]
		if status_log[params_fsps[pp]] == 0:
			sp.params[str_temp] = params_val[params_fsps[pp]]
		elif status_log[params_fsps[pp]] == 1:
			sp.params[str_temp] = math.pow(10.0,params_val[params_fsps[pp]])

	sp.params['imf_type'] = imf_type
	# gas phase metallicity:
	sp.params['gas_logz'] = params_val['logzsol']

	# CSP
	sp.params["sfh"] = 3

	# make grid of SFH:
	sfh_t, sfh_sfr = construct_SFH(sfh_form=sfh_form,t0=t0,tau=tau,alpha=alpha,
										beta=beta,age=age,formed_mass=formed_mass)

	if np.isnan(sfh_sfr).any()==True or np.isinf(sfh_sfr).any()==True:
		idx_excld = np.where((np.isnan(sfh_sfr)==True) | (np.isinf(sfh_sfr)==True))
		sfh_sfr[idx_excld[0]] = 0.0

		if np.any(sfh_sfr > 1.0e-33) == False:
			if len(idx_excld[0]) <= len(sfh_sfr) - 3:
				scale_sfr = 1.0/max(sfh_sfr)
				sfh_sfr = sfh_sfr*scale_sfr

				sp.set_tabular_sfh(sfh_t, sfh_sfr)

				mass0 = sp.stellar_mass
				dust_mass0 = sp.dust_mass
				norm = formed_mass/mass0
				dust_mass = dust_mass0*norm/scale_sfr
				SFR = sp.sfr/scale_sfr
			else:
				SFR = 1.0e-33
				mass = formed_mass
				wave = np.zeros(5994)
				spec = np.zeros(5994)
				dust_mass = 1.0e-33
		else:
			sp.set_tabular_sfh(sfh_t, sfh_sfr)

			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass
			norm = formed_mass/mass0
			dust_mass = dust_mass0*norm
			SFR = sp.sfr
	else:
		if np.any(sfh_sfr > 1.0e-33) == False:
			scale_sfr = 1.0/max(sfh_sfr)
			sfh_sfr = sfh_sfr*scale_sfr

			sp.set_tabular_sfh(sfh_t, sfh_sfr)

			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass
			norm = formed_mass/mass0
			dust_mass = dust_mass0*norm/scale_sfr
			SFR = sp.sfr/scale_sfr

		else:
			sp.set_tabular_sfh(sfh_t, sfh_sfr)

			mass0 = sp.stellar_mass
			dust_mass0 = sp.dust_mass
			norm = formed_mass/mass0
			dust_mass = dust_mass0*norm
			SFR = sp.sfr

	return SFR, dust_mass


def get_dust_mass_mainSFH_fit(sp=None,imf_type=1,sfh_form='delayed_tau_sfh',params_fsps=['logzsol', 'log_tau', 'log_age', 
	'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 'log_qpah','log_fagn', 'log_tauagn'], params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):

	params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
	status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

	# get stellar mass:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	tau = math.pow(10.0,params_val['log_tau'])
	age = math.pow(10.0,params_val['log_age'])

	# input model parameters to FSPS:
	nparams_fsps = len(params_fsps)
	for pp in range(0,nparams_fsps):
		str_temp = params_assoc_fsps[params_fsps[pp]]
		if status_log[params_fsps[pp]] == 0:
			sp.params[str_temp] = params_val[params_fsps[pp]]
		elif status_log[params_fsps[pp]] == 1:
			sp.params[str_temp] = math.pow(10.0,params_val[params_fsps[pp]])

	sp.params['imf_type'] = imf_type
	# gas phase metallicity:
	sp.params['gas_logz'] = params_val['logzsol']

	# generate the SED:
	mass = sp.stellar_mass
	dust_mass0 = sp.dust_mass   ## in solar mass/norm

	# normalize:
	norm0 = formed_mass/mass
	dust_mass = dust_mass0*norm0

	#print ("mass0=%e  dust_mass0=%e formed_mass=%e  norm=%e  dust_mass=%e" % (mass,dust_mass0,formed_mass,norm0,dust_mass))

	return dust_mass


def get_dust_mass_fagnbol_mainSFH_fit(sp=None,imf_type=1,sfh_form='delayed_tau_sfh',params_fsps=['logzsol', 'log_tau', 'log_age', 
	'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 'log_qpah','log_fagn', 'log_tauagn'], params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):

	params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
	status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

	# get stellar mass:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	tau = math.pow(10.0,params_val['log_tau'])
	age = math.pow(10.0,params_val['log_age'])

	# input model parameters to FSPS:
	nparams_fsps = len(params_fsps)
	for pp in range(0,nparams_fsps):
		str_temp = params_assoc_fsps[params_fsps[pp]]
		if status_log[params_fsps[pp]] == 0:
			sp.params[str_temp] = params_val[params_fsps[pp]]
		elif status_log[params_fsps[pp]] == 1:
			sp.params[str_temp] = math.pow(10.0,params_val[params_fsps[pp]])

	sp.params['imf_type'] = imf_type
	# gas phase metallicity:
	sp.params['gas_logz'] = params_val['logzsol']

	# generate the SED and get AGN luminosity
	wave, spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
	lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

	sp.params["fagn"] = 0.0
	wave, spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
	lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=spec)

	fagn_bol = (lbol_agn-lbol_noagn)/lbol_agn

	# get dust mass:
	mass = sp.stellar_mass
	dust_mass0 = sp.dust_mass   ## in solar mass/norm

	# normalize:
	norm0 = formed_mass/mass
	dust_mass = dust_mass0*norm0

	#print ("mass0=%e  dust_mass0=%e formed_mass=%e  norm=%e  dust_mass=%e" % (mass,dust_mass0,formed_mass,norm0,dust_mass))

	return dust_mass, fagn_bol


def spec_given_SFH_ZH(lbt=[],SFH_sfr=[],ZH_logzsol=[],z=0.001,cosmo='flat_LCDM',H0=70.0,Om0=0.3,imf=1,duste_switch='noduste',
	add_neb_emission=1,dust_ext_law='Cal2000',add_agn=0,add_igm_absorption=0,igm_type=1,dust1=0.0,dust2=0.0,dust_index=-99.0,
	gas_logu=-2.0,log_gamma=-99.0,log_umin=-99.0,log_qpah=-99.0,log_fagn=-99.0,log_tauagn=-99.0):
	"""A function to calculate spectra of a galaxy given SFH and metal enrichment history. SFH_sfr is in unit of M_solar/yr.
	lbt should be in Gyr. 
	"""
	
	# calling FSPS:
	sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)

	sp.params['imf_type'] = imf

	# dust emission:
	if duste_switch == 'duste' or duste_switch == 1:
		sp.params["add_dust_emission"] = True
		sp.params["duste_gamma"] = math.pow(10.0,log_gamma) 
		sp.params["duste_umin"] = math.pow(10.0,log_umin)
		sp.params["duste_qpah"] = math.pow(10.0,log_qpah)
	elif duste_switch == 'noduste' or duste_switch == 0:
		sp.params["add_dust_emission"] = False
	# nebular emission:
	if add_neb_emission == 1:
		sp.params["add_neb_emission"] = True
		sp.params['gas_logu'] = -2.0
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False
	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,log_fagn)
		sp.params["agn_tau"] = math.pow(10.0,log_tauagn)
	elif add_agn == 0:
		sp.params["fagn"] = 0
	# SSP
	sp.params["sfh"] = 0
	# dust attenuation:
	if dust_ext_law=='CF2000' or dust_ext_law==0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		sp.params["dust_index"] = dust_index
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
		sp.params["dust1"] = dust1
		sp.params["dust2"] = dust2
	elif dust_ext_law=='Cal2000' or dust_ext_law==1:
		sp.params["dust_type"] = 2
		sp.params["dust1"] = 0
		sp.params["dust2"] = dust2

	# iteration:
	ntimes = len(lbt)
	spec_array = []
	survive_mass = 0
	for tt in range(0,ntimes-1):
		# calculate mass formed:
		formed_mass = abs(0.5*(lbt[tt]-lbt[tt+1])*(SFH_sfr[tt]+SFH_sfr[tt+1]))*1.0e+9
		age0 = 0.5*(lbt[tt]+lbt[tt+1])

		ave_Z = 0.5*(ZH_logzsol[tt]+ZH_logzsol[tt+1])
		sp.params["logzsol"] = ave_Z
		sp.params['gas_logz'] = ave_Z
		sp.params['tage'] = age0

		wave, spec0 = sp.get_spectrum(peraa=True,tage=age0) ## spectrum in L_sun/AA
		mass0 = sp.stellar_mass
		
		spec_array.append(spec0*formed_mass)
		survive_mass = survive_mass + (mass0*formed_mass)

		# end of for tt: ntimes

	spec = np.sum(spec_array, axis=0)

	# redshifting:
	#spec_wave,spec_flux = redshifting.cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=z,wave=wave,spec=spec) ### in erg/s/cm^2/Ang.
	spec_wave,spec_flux = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=z,wave=wave,spec=spec) ### in erg/s/cm^2/Ang.

	return spec_wave,spec_flux,survive_mass


def spec_given_ages_mass_Z(grid_age=[],grid_mass=[],grid_logzsol=[],z=0.001,cosmo='flat_LCDM',H0=70.0,Om0=0.3,imf=1,duste_switch='noduste',
	add_neb_emission=1,dust_ext_law='Cal2000',add_agn=0,add_igm_absorption=0,igm_type=1,dust1=0.0,dust2=0.0,dust_index=-99.0,gas_logu=-2.0,
	log_gamma=-99.0,log_umin=-99.0,log_qpah=-99.0,log_fagn=-99.0,log_tauagn=-99.0):
	"""A function for generating model spectrum of a galaxy given input of ages, mass, and Z
	"""

	# calling FSPS:
	sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)

	sp.params['imf_type'] = imf

	# dust emission switch:
	if duste_switch == 'duste' or duste_switch == 1:
		sp.params["add_dust_emission"] = True
		sp.params["duste_gamma"] = math.pow(10.0,log_gamma) 
		sp.params["duste_umin"] = math.pow(10.0,log_umin)
		sp.params["duste_qpah"] = math.pow(10.0,log_qpah)
	elif duste_switch == 'noduste' or duste_switch == 0:
		sp.params["add_dust_emission"] = False
	# nebular emission switch:
	if add_neb_emission == 1:
		sp.params["add_neb_emission"] = True
		sp.params['gas_logu'] = -2.0
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False
	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,log_fagn)
		sp.params["agn_tau"] = math.pow(10.0,log_tauagn)
	elif add_agn == 0:
		sp.params["fagn"] = 0
	# SSP
	sp.params["sfh"] = 0
	# dust attenuation:
	if dust_ext_law=='CF2000' or dust_ext_law==0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		sp.params["dust_index"] = dust_index
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
		sp.params["dust1"] = dust1
		sp.params["dust2"] = dust2
	elif dust_ext_law=='Cal2000' or dust_ext_law==1:
		sp.params["dust_type"] = 2
		sp.params["dust1"] = 0
		sp.params["dust2"] = dust2

	# iteration:
	ntimes = len(grid_age)
	spec_array = []
	for tt in range(0,ntimes):
		sp.params["logzsol"] = grid_logzsol[tt]
		sp.params['gas_logz'] = grid_logzsol[tt]
		sp.params['tage'] = grid_age[tt]

		wave, spec0 = sp.get_spectrum(peraa=True,tage=grid_age[tt]) ## spectrum in L_sun/AA
		mass0 = sp.stellar_mass

		norm = grid_mass[tt]/mass0
		spec_array.append(spec0*norm)

		# end of for tt: ntimes

	spec = np.sum(spec_array, axis=0)

	# redshifting:
	#spec_wave,spec_flux = redshifting.cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=z,wave=wave,spec=spec) ### in erg/s/cm^2/Ang.
	spec_wave,spec_flux = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=z,wave=wave,spec=spec) ### in erg/s/cm^2/Ang.

	return spec_wave,spec_flux


def convert_unit_spec_from_ergscm2A(wave,spec,funit='Jy'):
	"""A function to convert unit of flux from 'erg/s/cm2/A' --> 'erg/s/cm2' or 'Jy'
	"""
	if funit=='erg/s/cm2/A' or funit==0:
		spec_new = spec
	elif funit=='erg/s/cm2' or funit==1:
		spec_new = np.asarray(spec)*np.asarray(wave)
	elif funit=='Jy' or funit==2:
		spec_new = np.asarray(spec)*np.asarray(wave)*np.asarray(wave)/1.0e-23/2.998e+18
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	return spec_new


def get_nebem_wave(z):
	"""A function to get list of nebular emission wavelengths
	"""

	sp = fsps.StellarPopulation(zcontinuous=1,add_neb_emission=1)

	sp.params['logzsol'] = -1.0
	sp.params['gas_logz'] = -1.0
	sp.params['gas_logu'] = -2.5
	wave, spec = sp.get_spectrum(tage=0.001)

	nebem_wave0 = sp.emline_wavelengths
	nebem_wave = nebem_wave0*(1.0+z)

	return nebem_wave

def get_no_nebem_wave(z,wave,del_wave):
	"""A function to get list of wavelengths (from the input array of wavelengths) that don't have emission line.

	:param z:
		Redshift

	:param wave:
		List of wavelengths

	:param del_wave:
		Assumption for the half of the width of the emission lines. del_wave is in Ang. 
	"""
	sp = fsps.StellarPopulation(zcontinuous=1,add_neb_emission=1)

	sp.params['logzsol'] = -1.0
	sp.params['gas_logz'] = -1.0
	sp.params['gas_logu'] = -2.5
	wave0, spec = sp.get_spectrum(tage=0.001)

	nebem_wave0 = sp.emline_wavelengths
	nebem_wave = nebem_wave0*(1.0+z)
	n_nebem_wave = len(nebem_wave)

	nwave = len(wave)
	status_excld = np.zeros(nwave)

	for ii in range(0,n_nebem_wave):
		idx = np.where((wave >= (nebem_wave[ii]-del_wave)) & (wave <= (nebem_wave[ii]+del_wave)))
		for jj in range(0,len(idx[0])):
			status_excld[idx[0][jj]] = 1

	wave_clean0 = []
	wave_mask = np.zeros(nwave)
	for ii in range(0,nwave):
		if status_excld[ii] == 0:
			wave_clean0.append(wave[ii])
		elif status_excld[ii] == 1:
			wave_mask[ii] = 1

	wave_clean = np.asarray(wave_clean0)

	return wave_clean,wave_mask


def get_no_nebem_wave_fit(sp=None,z=None,wave=[],del_wave=10.0):
	"""A function to get list of wavelengths (from the input array of wavelengths) that don't have emission line.

	:param z:
		Redshift

	:param wave:
		List of wavelengths

	:param del_wave:
		Assumption for the half of the width of the emission lines. del_wave is in Ang. 
	"""
	sp.params["sfh"] = 4
	sp.params['logzsol'] = -1.0
	sp.params['gas_logz'] = -1.0
	sp.params['gas_logu'] = -2.5
	wave0, spec = sp.get_spectrum(tage=0.001)

	nebem_wave = sp.emline_wavelengths
	nebem_wave = nebem_wave*(1.0+z)

	min_wave = min(wave)
	max_wave = max(wave)

	idx = np.where((nebem_wave>min_wave-del_wave) & (nebem_wave<max_wave+del_wave))
	nebem_wave = nebem_wave[idx[0]]

	nwave = len(wave)
	flag_excld = np.zeros(nwave)
	for ii in range(0,len(nebem_wave)):
		idx_excld = np.where((wave>=nebem_wave[ii]-del_wave) & (wave<=nebem_wave[ii]+del_wave))
		flag_excld[idx_excld[0]] = 1

	idx_excld = np.where(flag_excld==1)
	wave_clean = np.delete(wave, idx_excld[0])
	waveid_excld = idx_excld[0]

	return wave_clean,waveid_excld


def calc_bollum_from_spec_rest(spec_wave=[],spec_lum=[]):
	""" Function for calculating bolometric luminosity of rest-frame model spectrum in L_sun/A.
	"""

	# integrate
	wave_left = spec_wave[0:len(spec_wave)-1]
	lum_left = spec_lum[0:len(spec_wave)-1]

	wave_right = spec_wave[1:len(spec_wave)]
	lum_right = spec_lum[1:len(spec_wave)]

	areas = 0.5*(lum_left+lum_right)*(wave_right-wave_left)

	bol_lum = np.sum(areas)										# in L_sun

	l_sun = 3.826e+33      										# in erg/s
	bol_lum = bol_lum*l_sun

	return bol_lum


### calculate bolometric luminosity from a given spectrum in flux per unit wavelength:
### flux_lambda in unit erg/s/cm^2/Ang. and bolometric luminosity in erg/s
def calc_bollum_from_spec(spec_wave=[],spec_flux=[],wave_min=1000,wave_max=10000,gal_z=0.01,cosmo='flat_LCDM',H0=70.0,Om0=0.3):
	"""A function for calculating bolometric luminosity from a given spectrum in flux per unit wavelength: flux_lambda in unit erg/s/cm^2/Ang.
	The output bolometric luminosity is in erg/s.
	"""

	from astropy.cosmology import FlatLambdaCDM, WMAP5, WMAP7, WMAP9, Planck13, Planck15  
	from astropy import units as u

	# get luminosity distance:
	if cosmo=='flat_LCDM' or cosmo==0:
		cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
		DL0 = cosmo1.luminosity_distance(gal_z)      # in unit of Mpc
	elif cosmo=='WMAP5' or cosmo==1:
		DL0 = WMAP5.luminosity_distance(gal_z)
	elif cosmo=='WMAP7' or cosmo==2:
		DL0 = WMAP7.luminosity_distance(gal_z)
	elif cosmo=='WMAP9' or cosmo==3:
		DL0 = WMAP9.luminosity_distance(gal_z)
	elif cosmo=='Planck13' or cosmo==4:
		DL0 = Planck13.luminosity_distance(gal_z)
	elif cosmo=='Planck15' or cosmo==5:
		DL0 = Planck15.luminosity_distance(gal_z)
	#elif cosmo=='Planck18' or cosmo==6:
	#	DL0 = Planck18.luminosity_distance(gal_z)
     
	DL = DL0.value
	#print ("DL=%e Mpc" % DL)
	DL_temp = DL*1.0e+6*u.parsec
	DL_cm0 = DL_temp.to(u.cm)
	DL_cm = DL_cm0.value                    # in unit cm

	spec_lum_lambda = 4.0*math.pi*DL_cm*DL_cm*spec_flux  # in unit of erg/s/Ang.

	# cut spectrum:
	idx_incld = np.where((spec_wave >= wave_min) & (spec_wave <= wave_max))
	id_min = min(idx_incld[0])
	id_max = max(idx_incld[0])
	cut_spec_wave = spec_wave[id_min:id_max]
	cut_spec_lum_lambda = spec_lum_lambda[id_min:id_max]

	# integrate:
	bol_lum = 0
	for ii in range(0,len(cut_spec_wave)-1):
		area_bin = 0.5*(cut_spec_lum_lambda[ii]+cut_spec_lum_lambda[ii+1])*(cut_spec_wave[ii+1]-cut_spec_wave[ii])
		bol_lum = bol_lum + area_bin

	return bol_lum


def breakdown_CSP_into_SSP(sp=None,imf_type=1,logzsol=0.0,CSP_age=None,SFH_lbt=[],SFH_SFR=[],del_t=0.001,dust_ext_law='Cal2000',
	dust_index=-0.7,duste_switch='noduste',add_neb_emission=1,add_agn=0,dust1=0,dust2=0,log_umin=0,log_qpah=0,log_gamma=0,
	log_fagn=0,log_tauagn=0,gal_z=0.001,cosmo='flat_LCDM',H0=70.0,Om0=0.3):
	"""A function to break down a CSP into its SSP components
	"""

	# break down the SFH:
	sfh_age = np.linspace(del_t,int(CSP_age/del_t)*del_t,int(CSP_age/del_t))
	f = interp1d(SFH_lbt, SFH_SFR, fill_value="extrapolate")
	sfh_sfr = f(sfh_age)
	sfh_sfr[0] = sfh_sfr[1]
	sfh_sfr[len(sfh_age)-1] = sfh_sfr[len(sfh_age)-2]

	# get CSP age:
	if CSP_age == None:
		CSP_age = max(SFH_lbt)

	#### ====================================================== #####
	# calling FSPS:
	if sp == None:
		sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type)

	# dust emission:
	if duste_switch == 'duste' or duste_switch == 1:
		sp.params["add_dust_emission"] = True
		sp.params["duste_gamma"] = math.pow(10.0,log_gamma) 
		sp.params["duste_umin"] = math.pow(10.0,log_umin)
		sp.params["duste_qpah"] = math.pow(10.0,log_qpah)
	elif duste_switch == 'noduste' or duste_switch == 0:
		sp.params["add_dust_emission"] = False
	# nebular emission:
	if add_neb_emission == 1:
		sp.params["add_neb_emission"] = True
		sp.params['gas_logu'] = -2.0
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False
	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,log_fagn)
		sp.params["agn_tau"] = math.pow(10.0,log_tauagn)
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# SSP
	sp.params["sfh"] = 0
	# dust attenuation:
	if dust_ext_law=='CF2000' or dust_ext_law==0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		sp.params["dust_index"] = dust_index
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
		sp.params["dust1"] = dust1
		sp.params["dust2"] = dust2
	elif dust_ext_law=='Cal2000' or dust_ext_law==1:
		sp.params["dust_type"] = 2
		sp.params["dust1"] = 0
		sp.params["dust2"] = dust2
	#### ====================================================== #####

	# get number of wavelength:
	sp.params["logzsol"] = logzsol
	sp.params["gas_logz"] = logzsol
	sp.params["tage"] = CSP_age
	wave, spec = sp.get_spectrum(peraa=True,tage=CSP_age) ## spectrum in L_sun/AA
	nwaves = len(wave)

	# get the spectra of SSPs:
	nSSPs = len(sfh_age)
	SSP_spectra = np.zeros((nSSPs,nwaves))
	SSP_wave = np.zeros(nwaves)
	for tt in range(0,nSSPs):
		sp.params["logzsol"] = logzsol
		sp.params["gas_logz"] = logzsol
		sp.params["tage"] = sfh_age[tt]
		wave, spec = sp.get_spectrum(peraa=True,tage=sfh_age[tt]) ## spectrum in L_sun/AA
		mass = sp.stellar_mass
		norm = sfh_sfr[tt]*del_t*1.0e+9/mass

		# redshifting:
		#spec_wave,spec_flux = redshifting.cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=spec*norm) ### in erg/s/cm^2/Ang.
		spec_wave,spec_flux = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=gal_z,wave=wave,spec=spec*norm) ### in erg/s/cm^2/Ang.

		SSP_spectra[tt] = spec_flux
		SSP_wave = spec_wave
		# end of for tt: nSSPs

	return SSP_wave,SSP_spectra,sfh_age,sfh_sfr



### define a function to break down a CSP into its SSP components:
def breakdown_CSP_into_SSP_restframe(sp=None,imf_type=1,logzsol=0.0,CSP_age=None,SFH_lbt=[],SFH_SFR=[],del_t=0.001,dust_ext_law='Cal2000',
	dust_index=-0.7,duste_switch='noduste',add_neb_emission=1,add_agn=0,dust1=0,dust2=0,log_umin=0,log_qpah=0,log_gamma=0,log_fagn=0,
	log_tauagn=0):
	"""A function to break down a CSP into its SSP components (in rest-frame)
	"""

	# break down the SFH:
	sfh_age = np.linspace(del_t,int(CSP_age/del_t)*del_t,int(CSP_age/del_t))
	f = interp1d(SFH_lbt, SFH_SFR, fill_value="extrapolate")
	sfh_sfr = f(sfh_age)
	sfh_sfr[0] = sfh_sfr[1]
	sfh_sfr[len(sfh_age)-1] = sfh_sfr[len(sfh_age)-2]

	# get CSP age:
	if CSP_age == None:
		CSP_age = max(SFH_lbt)

	#### ====================================================== #####
	# calling FSPS:
	if sp == None:
		sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type)

	# dust emission switch:
	if duste_switch == 'duste' or duste_switch == 1:
		sp.params["add_dust_emission"] = True
		sp.params["duste_gamma"] = math.pow(10.0,log_gamma) 
		sp.params["duste_umin"] = math.pow(10.0,log_umin)
		sp.params["duste_qpah"] = math.pow(10.0,log_qpah)
	elif duste_switch == 'noduste' or duste_switch == 0:
		sp.params["add_dust_emission"] = False
	# nebular emission switch:
	if add_neb_emission == 1:
		sp.params["add_neb_emission"] = True
		sp.params['gas_logu'] = -2.0
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False
	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,log_fagn)
		sp.params["agn_tau"] = math.pow(10.0,log_tauagn)
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# SSP
	sp.params["sfh"] = 0
	# dust attenuation:
	if dust_ext_law=='CF2000' or dust_ext_law==0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		sp.params["dust_index"] = dust_index
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
		sp.params["dust1"] = dust1
		sp.params["dust2"] = dust2
	elif dust_ext_law=='Cal2000' or dust_ext_law==1:
		sp.params["dust_type"] = 2
		sp.params["dust1"] = 0
		sp.params["dust2"] = dust2
	#### ====================================================== #####

	# get number of wavelength:
	sp.params["logzsol"] = logzsol
	sp.params["gas_logz"] = logzsol
	sp.params["tage"] = CSP_age
	wave, spec = sp.get_spectrum(peraa=True,tage=CSP_age) ## spectrum in L_sun/AA
	nwaves = len(wave)

	# get the spectra of SSPs:
	nSSPs = len(sfh_age)
	SSP_spectra = np.zeros((nSSPs,nwaves))
	SSP_wave = np.zeros(nwaves)
	for tt in range(0,nSSPs):
		sp.params["logzsol"] = logzsol
		sp.params["gas_logz"] = logzsol
		sp.params["tage"] = sfh_age[tt]
		wave, spec = sp.get_spectrum(peraa=True,tage=sfh_age[tt]) ## spectrum in L_sun/AA
		mass = sp.stellar_mass
		norm = sfh_sfr[tt]*del_t*1.0e+9/mass

		SSP_spectra[tt] = spec*norm
		SSP_wave = wave
		# end of for tt: nSSPs

	return SSP_wave,SSP_spectra,sfh_age,sfh_sfr


def get_emlines_luminosities(sp=None,emlines_rest_waves=[],imf_type=1,duste_switch=1,add_neb_emission=1,dust_ext_law='Cal2000',
	sfh_form='delayed_tau_sfh',add_agn=1,sfh_t=[],sfh_sfr=[],params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,
	'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,
	'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):
	"""A function to get luminosities of some emission lines.
	"""

	formed_mass = math.pow(10.0,params_val['log_mass'])

	# get some input properties:
	age = math.pow(10.0,params_val['log_age'])
	t0 = math.pow(10.0,params_val['log_t0'])
	tau = math.pow(10.0,params_val['log_tau'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	sp.params['imf_type'] = imf_type

	# dust emission:
	if duste_switch == 'duste' or duste_switch == 1:
		sp.params["add_dust_emission"] = True
		sp.params["duste_gamma"] = math.pow(10.0,params_val['log_gamma']) 
		sp.params["duste_umin"] = math.pow(10.0,params_val['log_umin'])
		sp.params["duste_qpah"] = math.pow(10.0,params_val['log_qpah'])
	elif duste_switch == 'noduste' or duste_switch == 0:
		sp.params["add_dust_emission"] = False
	# nebular emission:
	if add_neb_emission == 1:
		sp.params["add_neb_emission"] = True
		sp.params['gas_logu'] = -2.0
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False
	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,params_val['log_fagn'])
		sp.params["agn_tau"] = math.pow(10.0,params_val['log_tauagn'])
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# CSP
	sp.params["sfh"] = 3
	# dust attenuation:
	if dust_ext_law=='CF2000' or dust_ext_law==0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		sp.params["dust_index"] = params_val['dust_index']
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
		sp.params["dust1"] = params_val['dust1']
		sp.params["dust2"] = params_val['dust2']
	elif dust_ext_law=='Cal2000' or dust_ext_law==1:
		sp.params["dust_type"] = 2
		sp.params["dust1"] = 0
		sp.params["dust2"] = params_val['dust2']
	# other parameters:
	sp.params["logzsol"] = params_val['logzsol'] 
	sp.params['gas_logz'] = params_val['logzsol'] 
	sp.params['tage'] = age

	# make grid of SFH:
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==0 or sfh_form==1 or sfh_form==2 or sfh_form==3 or sfh_form==4:
		sfh_t, sfh_sfr = construct_SFH(sfh_form=sfh_form,t0=t0,tau=tau,alpha=alpha,
										beta=beta,age=age,formed_mass=formed_mass)
	elif sfh_form=='arbitrary_sfh':
		sfh_t0 = sfh_t
		sfh_sfr0 = sfh_sfr
		del_t = 0.001
		sfh_t = np.linspace(0,age,int(age/del_t)+1)
		f = interp1d(sfh_t0, sfh_sfr0, fill_value="extrapolate")
		sfh_sfr = f(sfh_t)
		idx_neg = np.where(sfh_sfr < 0)
		for ii in range(0,len(idx_neg[0])):
			sfh_sfr[idx_neg[0][ii]] = 0.0
	else:
		print ("SFH choice is not recognized!")
		sys.exit()

	if np.any(sfh_sfr > 1.0e-33) == False:
		if len(sfh_sfr) == 0:
			SFR = 1.0e-12
			mass = formed_mass
			wave = np.zeros(5994)
			spec = np.zeros(5994)
			dust_mass = formed_mass*1.0e-5
		elif np.isnan(sfh_sfr).any() == True:
			if np.isnan(sfh_sfr).all() == True: 
				SFR = 1.0e-12
				mass = formed_mass
				wave = np.zeros(5994)
				spec = np.zeros(5994)
				dust_mass = formed_mass*1.0e-5
			else:
				idx_excld = np.where(np.isnan(sfh_sfr) == True)
				sfh_t_new = np.delete(sfh_t,idx_excld[0])
				sfh_sfr_new = np.delete(sfh_sfr,idx_excld[0])

				sp.set_tabular_sfh(sfh_t_new, sfh_sfr_new)
				wave, spec0 = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
				
				# get emission lines:
				emline_lum = sp.emline_luminosity
				emline_wave = sp.emline_wavelengths


		else:
			scale_sfr = 1.0/max(sfh_sfr)
			sfh_sfr = sfh_sfr*scale_sfr

			sp.set_tabular_sfh(sfh_t, sfh_sfr) 
			wave, spec0 = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA

			# get emission lines:
			emline_lum = sp.emline_luminosity
			emline_wave = sp.emline_wavelengths
			

	else:
		sp.set_tabular_sfh(sfh_t, sfh_sfr)
		wave, spec0 = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA

		# get emission lines:
		emline_lum = sp.emline_luminosity
		emline_wave = sp.emline_wavelengths


	# select among the list of the emission lines:
	nlines = len(emlines_rest_waves)

	emline_waves = np.zeros(nlines)
	emline_luminosities = np.zeros(nlines)
	for ii in range(0,nlines):
		del_waves = np.abs(emline_wave - emlines_rest_waves[ii])
		idx, min_val = min(enumerate(del_waves), key=operator.itemgetter(1))
		emline_waves[ii] = emline_wave[idx]
		emline_luminosities[ii] = emline_lum[idx]*3.826e+33    ### in unit of erg/s

	return emline_waves,emline_luminosities



