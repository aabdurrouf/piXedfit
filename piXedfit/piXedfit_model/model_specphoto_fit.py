import numpy as np
import math
import sys, os
import random
import fsps
import operator
from astropy.io import fits
from scipy.interpolate import interp1d

from ..utils.redshifting import cosmo_redshifting
from ..utils.filtering import filtering, cwave_filters, filtering_match_filters_array
from ..utils.igm_absorption import igm_att_madau, igm_att_inoue
from .model_utils import *

# warning is not logged here. Perfect for clean unit test output
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0


__all__ = ["generate_modelSED_propspecphoto_fit", "generate_modelSED_spec_fit", "generate_modelSED_specphoto_fit", "generate_modelSED_propspecphoto_nomwage_fit"]


def generate_modelSED_propspecphoto_fit(sp=None,imf_type=1,sfh_form='delayed_tau_sfh',filters=['galex_fuv','galex_nuv','sdss_u',
	'sdss_g','sdss_r','sdss_i','sdss_z'],add_igm_absorption=0,igm_type=1,params_fsps=['logzsol', 'log_tau', 'log_age', 
	'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 'log_qpah','log_fagn', 'log_tauagn'], params_val={'log_mass':0.0,
	'z':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,'log_qpah':-99.0,'log_umin':-99.0,'log_gamma':-99.0,'dust1':-99.0,'dust2':-99.0,
	'dust_index':-99.0,'log_age':-99.0,'log_alpha':-99.0,'log_beta':-99.0,'log_t0':-99.0,'log_tau':-99.0,'logzsol':-99.0},
	DL_Gpc=0.0,cosmo='flat_LCDM',H0=70.0,Om0=0.3,free_z=0,trans_fltr_int=None):
	"""A function to generate model spectrophotometric SED in which outputs are: properties, spectrum, and photometric SED

	:param sp:
		Initialization of FSPS, such as sp=fsps.StellarPopulation()

	:param imf_type:
		Choice for the IMF. Choices are: [0:Salpeter(1955), 1:Chabrier(2003), 2:Kroupa(2001)]

	:param sfh_form:
		Choice for the parametric SFH model. 
		Options are: ['tau_sfh', 'delayed_tau_sfh', 'log_normal_sfh', 'gaussian_sfh', 'double_power_sfh']

	:param filters:
		A list of photometric filters.

	:param add_igm_absorption:
		Switch for the IGM absorption.

	:param igm_type:
		Choice for the IGM absorption model. Options are: [0/'madau1995':Madau(1995); 1/'inoue2014':Inoue+(2014)]

	:param cosmo (default: 'flat_LCDM'):
		Choices for the cosmological parameters. The choices are: ['flat_LCDM', 'WMAP5', 
		'WMAP7', 'WMAP9', 'Planck13', 'Planck15'], similar to the choices available in the 
		Astropy Cosmology package: https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies.
		If 'flat_LCDM' is chosen, the input H0 and Om0 should be provided.

	:param H0, Om0 (default: H0=70.0, Om0=0.3):
		Hubble constant and Omega matter at z=0.0. 

	:param param_val:
		A dictionary of parameters values.
	"""
	
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

	# generate the SED:
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form==0 or sfh_form==1:
		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		# get model mass:
		mass = sp.stellar_mass
		# get dust mass: 
		dust_mass0 = sp.dust_mass   # in solar mass/norm
	elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==2 or sfh_form==3 or sfh_form==4:
		SFR_fSM,mass,wave,extnc_spec,dust_mass0 = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)

	# redshifting
	#redsh_wave,redsh_spec0 = redshifting.cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)
	redsh_wave,redsh_spec0 = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	if add_igm_absorption == 1:
		if igm_type == 0 or igm_type == 'madau1995':
			trans = igm_att_madau(redsh_wave,params_val['z'])
			temp = redsh_spec0
			redsh_spec0 = temp*trans
		elif igm_type == 1 or igm_type == 'inoue2014':
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			temp = redsh_spec0
			redsh_spec0 = temp*trans

	# normalize:
	norm0 = formed_mass/mass
	redsh_spec = redsh_spec0*norm0
	dust_mass = dust_mass0*norm0

	# filtering:
	if free_z == 1:
		photo_SED_flux = filtering(redsh_wave,redsh_spec,filters) ##########
	elif free_z == 0:
		photo_SED_flux = filtering_match_filters_array(redsh_wave,redsh_spec,filters,trans_fltr_int)
	# get central wavelength of all filters:
	photo_cwave = cwave_filters(filters)

	# calculate SFR:
	SFR_exp = 1.0/np.exp(age/tau)
	if sfh_form=='tau_sfh' or sfh_form==0:
		SFR_fSM = formed_mass*SFR_exp/tau/(1.0-SFR_exp)/1e+9
	elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
		SFR_fSM = formed_mass*age*SFR_exp/((tau*tau)-((age*tau)+(tau*tau))*SFR_exp)/1e+9
	# calculate mw-age:
	mw_age = calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,
				age=age,formed_mass=formed_mass)

	# make arrays for outputs:
	SED_prop = {}
	SED_prop['SM'] = formed_mass
	SED_prop['survive_mass'] = mass
	SED_prop['SFR'] = SFR_fSM
	SED_prop['mw_age'] = mw_age
	SED_prop['dust_mass'] = dust_mass

	spec_SED = {}
	spec_SED['wave'] = redsh_wave
	spec_SED['flux'] = redsh_spec

	photo_SED = {}
	photo_SED['wave'] = photo_cwave
	photo_SED['flux'] = photo_SED_flux

	return SED_prop,photo_SED,spec_SED


def generate_modelSED_spec_fit(sp=None,imf_type=1,sfh_form='delayed_tau_sfh',add_igm_absorption=0,igm_type=1,
	params_fsps=['logzsol', 'log_tau', 'log_age', 'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 
	'log_qpah','log_fagn', 'log_tauagn'], params_val={'log_mass':0.0,'z':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,
	'log_qpah':-99.0,'log_umin':-99.0,'log_gamma':-99.0,'dust1':-99.0,'dust2':-99.0,'dust_index':-99.0,'log_age':-99.0,
	'log_alpha':-99.0,'log_beta':-99.0,'log_t0':-99.0,'log_tau':-99.0,'logzsol':-99.0},DL_Gpc=0.0,cosmo='flat_LCDM',
	H0=70.0,Om0=0.3):
	"""A function to generate model spectroscopic SED

	:param sp:
		Initialization of FSPS, such as sp=fsps.StellarPopulation()

	:param imf_type:
		Choice for the IMF. Choices are: [0:Salpeter(1955), 1:Chabrier(2003), 2:Kroupa(2001)]

	:param sfh_form:
		Choice for the parametric SFH model. 
		Options are: ['tau_sfh', 'delayed_tau_sfh', 'log_normal_sfh', 'gaussian_sfh', 'double_power_sfh']

	:param add_igm_absorption:
		Switch for the IGM absorption.

	:param igm_type:
		Choice for the IGM absorption model. Options are: [0/'madau1995':Madau(1995); 1/'inoue2014':Inoue+(2014)]

	:param cosmo (default: 'flat_LCDM'):
		Choices for the cosmological parameters. The choices are: ['flat_LCDM', 'WMAP5', 
		'WMAP7', 'WMAP9', 'Planck13', 'Planck15'], similar to the choices available in the 
		Astropy Cosmology package: https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies.
		If 'flat_LCDM' is chosen, the input H0 and Om0 should be provided.

	:param H0, Om0 (default: H0=70.0, Om0=0.3):
		Hubble constant and Omega matter at z=0.0. 

	:param param_val:
		A dictionary of parameters values.
	"""
	
	params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
	status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

	# get stellar mass:
	formed_mass = math.pow(10.0,params_val['log_mass'])

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
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form==0 or sfh_form==1:
		age = math.pow(10.0,params_val['log_age'])
		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		# get model mass:
		mass = sp.stellar_mass
		# get dust mass: 
		dust_mass0 = sp.dust_mass   ## in solar mass/norm
	elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==2 or sfh_form==3 or sfh_form==4:
		t0 = math.pow(10.0,params_val['log_t0'])
		tau = math.pow(10.0,params_val['log_tau'])
		age = math.pow(10.0,params_val['log_age'])
		alpha = math.pow(10.0,params_val['log_alpha'])
		beta = math.pow(10.0,params_val['log_beta'])
		SFR_fSM,mass,wave,extnc_spec,dust_mass0 = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)

	# redshifting
	#redsh_wave,redsh_spec0 = redshifting.cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)
	redsh_wave,redsh_spec0 = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	if add_igm_absorption == 1:
		if igm_type==0 or igm_type=='madau1995':
			trans = igm_att_madau(redsh_wave,params_val['z'])
			temp = redsh_spec0
			redsh_spec0 = temp*trans
		elif igm_type==1 or igm_type=='inoue2014':
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			temp = redsh_spec0
			redsh_spec0 = temp*trans

	# normalize:
	norm0 = formed_mass/mass
	redsh_spec = redsh_spec0*norm0

	spec_SED = {}
	spec_SED['wave'] = redsh_wave
	spec_SED['flux'] = redsh_spec

	return spec_SED



def generate_modelSED_specphoto_fit(sp=None,imf_type=1,sfh_form='delayed_tau_sfh',filters=['galex_fuv','galex_nuv','sdss_u',
	'sdss_g','sdss_r','sdss_i','sdss_z'],add_igm_absorption=0,igm_type=1,params_fsps=['logzsol', 'log_tau', 'log_age', 
	'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 'log_qpah','log_fagn', 'log_tauagn'], params_val={'log_mass':0.0,
	'z':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,'log_qpah':-99.0,'log_umin':-99.0,'log_gamma':-99.0,'dust1':-99.0,'dust2':-99.0,
	'dust_index':-99.0,'log_age':-99.0,'log_alpha':-99.0,'log_beta':-99.0,'log_t0':-99.0,'log_tau':-99.0,'logzsol':-99.0},
	DL_Gpc=0.0,cosmo='flat_LCDM',H0=70.0,Om0=0.3,free_z=0,trans_fltr_int=None):
	"""A function to generate model spectrophotometric SED

	:param sp:
		Initialization of FSPS, such as sp=fsps.StellarPopulation()

	:param imf_type:
		Choice for the IMF. Choices are: [0:Salpeter(1955), 1:Chabrier(2003), 2:Kroupa(2001)]

	:param sfh_form:
		Choice for the parametric SFH model. 
		Options are: ['tau_sfh', 'delayed_tau_sfh', 'log_normal_sfh', 'gaussian_sfh', 'double_power_sfh']

	:param filters:
		A list of photometric filters.

	:param add_igm_absorption:
		Switch for the IGM absorption.

	:param igm_type:
		Choice for the IGM absorption model. Options are: [0/'madau1995':Madau(1995); 1/'inoue2014':Inoue+(2014)]

	:param cosmo (default: 'flat_LCDM'):
		Choices for the cosmological parameters. The choices are: ['flat_LCDM', 'WMAP5', 
		'WMAP7', 'WMAP9', 'Planck13', 'Planck15'], similar to the choices available in the 
		Astropy Cosmology package: https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies.
		If 'flat_LCDM' is chosen, the input H0 and Om0 should be provided.

	:param H0, Om0 (default: H0=70.0, Om0=0.3):
		Hubble constant and Omega matter at z=0.0. 

	:param param_val:
		A dictionary of parameters values.
	"""
	
	params_assoc_fsps = {'logzsol':"logzsol", 'log_tau':"tau", 'log_age':"tage", 
					'dust_index':"dust_index", 'dust1':"dust1", 'dust2':"dust2",
					'log_gamma':"duste_gamma", 'log_umin':"duste_umin", 
					'log_qpah':"duste_qpah",'log_fagn':"fagn", 'log_tauagn':"agn_tau"}
	status_log = {'logzsol':0, 'log_tau':1, 'log_age':1, 'dust_index':0, 'dust1':0, 'dust2':0,
				'log_gamma':1, 'log_umin':1, 'log_qpah':1,'log_fagn':1, 'log_tauagn':1}

	# get stellar mass:
	formed_mass = math.pow(10.0,params_val['log_mass'])

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
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form==0 or sfh_form==1:
		age = math.pow(10.0,params_val['log_age'])
		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		# get model mass:
		mass = sp.stellar_mass
		# get dust mass: 
		dust_mass0 = sp.dust_mass   ## in solar mass/norm
	elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==2 or sfh_form==3 or sfh_form==4:
		t0 = math.pow(10.0,params_val['log_t0'])
		tau = math.pow(10.0,params_val['log_tau'])
		age = math.pow(10.0,params_val['log_age'])
		alpha = math.pow(10.0,params_val['log_alpha'])
		beta = math.pow(10.0,params_val['log_beta'])
		SFR_fSM,mass,wave,extnc_spec,dust_mass0 = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)

	# redshifting
	redsh_wave,redsh_spec0 = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	if add_igm_absorption == 1:
		if igm_type==0 or igm_type=='madau1995':
			trans = igm_att_madau(redsh_wave,params_val['z'])
			temp = redsh_spec0
			redsh_spec0 = temp*trans
		elif igm_type==1 or igm_type=='inoue2014':
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			temp = redsh_spec0
			redsh_spec0 = temp*trans

	# normalize:
	norm0 = formed_mass/mass
	redsh_spec = redsh_spec0*norm0
	dust_mass = dust_mass0*norm0

	# filtering:
	if free_z == 1:
		photo_SED_flux = filtering(redsh_wave,redsh_spec,filters) ##########
	elif free_z == 0:
		photo_SED_flux = filtering_match_filters_array(redsh_wave,redsh_spec,filters,trans_fltr_int)
	# get central wavelength of all filters:
	photo_cwave = cwave_filters(filters)

	spec_SED = {}
	spec_SED['spec_wave'] = redsh_wave
	spec_SED['spec_flux'] = redsh_spec

	photo_SED = {}
	photo_SED['photo_wave'] = photo_cwave
	photo_SED['photo_flux'] = photo_SED_flux

	return spec_SED,photo_SED



def generate_modelSED_propspecphoto_nomwage_fit(sp=None,imf_type=1,sfh_form='delayed_tau_sfh',filters=['galex_fuv','galex_nuv','sdss_u',
	'sdss_g','sdss_r','sdss_i','sdss_z'],add_igm_absorption=0,igm_type=1,params_fsps=['logzsol', 'log_tau', 'log_age', 
	'dust_index', 'dust1', 'dust2', 'log_gamma', 'log_umin', 'log_qpah','log_fagn', 'log_tauagn'], params_val={'log_mass':0.0,
	'z':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,'log_qpah':-99.0,'log_umin':-99.0,'log_gamma':-99.0,'dust1':-99.0,'dust2':-99.0,
	'dust_index':-99.0,'log_age':-99.0,'log_alpha':-99.0,'log_beta':-99.0,'log_t0':-99.0,'log_tau':-99.0,'logzsol':-99.0},
	DL_Gpc=0.0,cosmo='flat_LCDM',H0=70.0,Om0=0.3,free_z=0,trans_fltr_int=None):
	
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

	# generate the SED:
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form==0 or sfh_form==1:
		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		# get model mass:
		mass = sp.stellar_mass
		# get dust mass: 
		dust_mass0 = sp.dust_mass   # in solar mass/norm
	elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==2 or sfh_form==3 or sfh_form==4:
		SFR_fSM,mass,wave,extnc_spec,dust_mass0 = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)

	# redshifting
	#redsh_wave,redsh_spec0 = redshifting.cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)
	redsh_wave,redsh_spec0 = cosmo_redshifting(DL_Gpc=DL_Gpc,cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	if add_igm_absorption == 1:
		if igm_type==0 or igm_type=='madau1995':
			trans = igm_att_madau(redsh_wave,params_val['z'])
			temp = redsh_spec0
			redsh_spec0 = temp*trans
		elif igm_type==1 or igm_type=='inoue2014':
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			temp = redsh_spec0
			redsh_spec0 = temp*trans

	# normalize:
	norm0 = formed_mass/mass
	redsh_spec = redsh_spec0*norm0
	dust_mass = dust_mass0*norm0

	# filtering:
	if free_z == 1:
		photo_SED_flux = filtering(redsh_wave,redsh_spec,filters) ##########
	elif free_z == 0:
		photo_SED_flux = filtering_match_filters_array(redsh_wave,redsh_spec,filters,trans_fltr_int)
	# get central wavelength of all filters:
	photo_cwave = cwave_filters(filters)

	# calculate SFR:
	SFR_exp = 1.0/np.exp(age/tau)
	if sfh_form=='tau_sfh' or sfh_form==0:
		SFR_fSM = formed_mass*SFR_exp/tau/(1.0-SFR_exp)/1e+9
	elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
		SFR_fSM = formed_mass*age*SFR_exp/((tau*tau)-((age*tau)+(tau*tau))*SFR_exp)/1e+9

	# make arrays for outputs:
	SED_prop = {}
	SED_prop['SM'] = formed_mass
	SED_prop['survive_mass'] = mass
	SED_prop['SFR'] = SFR_fSM
	SED_prop['dust_mass'] = dust_mass

	photo_SED = {}
	photo_SED['wave'] = photo_cwave
	photo_SED['flux'] = photo_SED_flux

	spec_SED = {}
	spec_SED['wave'] = redsh_wave
	spec_SED['flux'] = redsh_spec

	return SED_prop,photo_SED,spec_SED

