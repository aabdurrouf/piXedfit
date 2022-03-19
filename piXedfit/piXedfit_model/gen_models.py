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

global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']


__all__ = ["generate_modelSED_propspecphoto", "generate_modelSED_spec", "generate_modelSED_photo", "generate_modelSED_specphoto", 
			"generate_modelSED_spec_decompose", "generate_modelSED_specphoto_decompose","save_models", "save_models_gridz_params", 
			"save_models_gridz_sed", "save_models_gridz_merge", "generate_mockSED", "add_fagn_bol_samplers"]

def generate_modelSED_propspecphoto(sp=None,imf_type=1,duste_switch=1,add_neb_emission=1,dust_ext_law='Cal2000',
	sfh_form='delayed_tau_sfh',add_agn=0,filters=['galex_fuv','galex_nuv','sdss_u','sdss_g','sdss_r','sdss_i','sdss_z'],
	add_igm_absorption=0,igm_type='inoue2014',cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0,params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):
	"""A function to generate model SED in which the output includes: properties, spectrum, and photometric fluxes

	:param sp:
		Initialization of FSPS, such as sp=fsps.StellarPopulation()

	:param imf_type:
		Choice for the IMF. Choices are: [0:Salpeter(1955), 1:Chabrier(2003), 2:Kroupa(2001)]

	:param duste_switch:
		Choice for turning on (1) or off (0) the dust emission modeling

	:param add_neb_emission:
		Choice for turning on (1) or off (0) the nebular emission modeling

	:param dust_ext_law:
		Choice of dust attenuation law. Options are: ['CF2000':Charlot&Fall(2000), 'Cal2000':Calzetti+2000]

	:param sfh_form:
		Choice for the parametric SFH model. 
		Options are: ['tau_sfh', 'delayed_tau_sfh', 'log_normal_sfh', 'gaussian_sfh', 'double_power_sfh', 'arbitrary_sfh']

	:param add_agn:
		Choice for turning on (1) or off (0) the AGN dusty torus modeling

	:param filters:
		List of photometric filters.

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
		Hubble constant and Omega matter at z=0.0

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param sfh_t, sfh_sfr:
		arrays for arbitrary SFH. These parameters only relevant if sfh_form='arbitrary_sfh'. 

	:param param_val:
		A dictionary of parameters values.
	"""
	
	# input parameters:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	age = math.pow(10.0,params_val['log_age'])
	tau = math.pow(10.0,params_val['log_tau'])
	t0 = math.pow(10.0,params_val['log_t0'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	if sp == None:
		sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type)

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
		sp.params['gas_logu'] = gas_logu
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False

	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,params_val['log_fagn'])
		sp.params["agn_tau"] = math.pow(10.0,params_val['log_tauagn'])
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# other parameters:
	sp.params["logzsol"] = params_val['logzsol'] 
	sp.params['gas_logz'] = params_val['logzsol']
	sp.params['tage'] = age

	# generate the SED:
	SFR_fSM = -99.0			# for temporary
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form==0 or sfh_form==1:
		if sfh_form=='tau_sfh' or sfh_form==0:
			sp.params["sfh"] = 1
		elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
			sp.params["sfh"] = 4
		sp.params["const"] = 0
		sp.params["sf_start"] = 0
		sp.params["sf_trunc"] = 0
		sp.params["fburst"] = 0
		sp.params["tburst"] = 30.0
		sp.params["tau"] = tau
		#sp.params["tage"] = age

		# dust atenuation:
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

		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		mass = sp.stellar_mass
		dust_mass0 = sp.dust_mass   # in solar mass/norm

		# total bolometric luminosity including AGN
		lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave, spec_lum=extnc_spec)

		# bolometric luminosity excluding AGN
		sp.params["fagn"] = 0.0
		wave9, spec9 = sp.get_spectrum(peraa=True,tage=age) 			# spectrum in L_sun/AA
		lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave9,spec_lum=spec9)

		# get fraction of AGN luminosity from the total bolometric luminosity
		fagn_bol = (lbol_agn-lbol_noagn)/lbol_agn
		log_fagn_bol = np.log10(fagn_bol)

		# normalization:
		norm0 = formed_mass/mass
		extnc_spec = extnc_spec*norm0
		dust_mass = dust_mass0*norm0

		# calculate SFR:
		SFR_exp = 1.0/np.exp(age/tau)
		if sfh_form=='tau_sfh' or sfh_form==0:
			SFR_fSM = formed_mass*SFR_exp/tau/(1.0-SFR_exp)/1e+9
		elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
			SFR_fSM = formed_mass*age*SFR_exp/((tau*tau)-((age*tau)+(tau*tau))*SFR_exp)/1e+9

	elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==2 or sfh_form==3 or sfh_form==4:
		sp.params["sfh"] = 3
		# dust atenuation:
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

		SFR_fSM,mass,wave,extnc_spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)

		# total bolometric luminosity including AGN
		lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=extnc_spec)

		# bolometric luminosity excluding AGN
		sp.params["fagn"] = 0.0		
		SFR_fSM9,mass9,wave9,spec9,dust_mass9 = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
		lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave9,spec_lum=spec9)

		# get fraction of AGN luminosity from the total bolometric luminosity
		fagn_bol = (lbol_agn-lbol_noagn)/lbol_agn
		log_fagn_bol = np.log10(fagn_bol)


	# redshifting
	#redsh_wave,redsh_spec0 = redshifting.cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)
	redsh_wave,redsh_spec0 = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	redsh_spec = redsh_spec0
	if add_igm_absorption == 1:
		if igm_type==0 or igm_type=='madau1995':
			trans = igm_att_madau(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans
		elif igm_type==1 or igm_type=='inoue2014':
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans

	# filtering:
	#photo_SED_flux = filtering.filtering(redsh_wave,redsh_spec,filters)
	photo_SED_flux = filtering(redsh_wave,redsh_spec,filters)
	
	# get central wavelength of all filters:
	#photo_cwave = filtering.cwave_filters(filters)
	photo_cwave = cwave_filters(filters)

	# calculate SFR:
	#SFR_exp = 1.0/np.exp(age/tau)
	#if sfh_form=='tau_sfh' or sfh_form==0:
	#	SFR_fSM = formed_mass*SFR_exp/tau/(1.0-SFR_exp)/1e+9
	#elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
	#	SFR_fSM = formed_mass*age*SFR_exp/((tau*tau)-((age*tau)+(tau*tau))*SFR_exp)/1e+9

	# calculate mw-age:
	mw_age = calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,
				age=age,formed_mass=formed_mass)

	# outputs:
	SED_prop = {}
	SED_prop['SM'] = formed_mass
	#SED_prop['survive_mass'] = mass
	SED_prop['SFR'] = SFR_fSM
	SED_prop['mw_age'] = mw_age
	SED_prop['dust_mass'] = dust_mass
	SED_prop['log_fagn_bol'] = log_fagn_bol

	spec_SED = {}
	spec_SED['wave'] = redsh_wave
	spec_SED['flux'] = redsh_spec

	photo_SED = {}
	photo_SED['wave'] = photo_cwave
	photo_SED['flux'] = photo_SED_flux

	return SED_prop, photo_SED, spec_SED



def generate_modelSED_spec(sp=None,imf_type=1,duste_switch=1,add_neb_emission=1,dust_ext_law='Cal2000',sfh_form='delayed_tau_sfh',
	add_agn=0,add_igm_absorption=0,igm_type='inoue2014',cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0,params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):

	"""Function for generating a model spectroscopic SED given some parameters.

	:param sp: (optional, default: None)
		Initialization of FSPS, such as `sp=fsps.StellarPopulation()`.

	:param imf_type: (default: 1)
		Choice for the IMF. Choices are: (1)0 for Salpeter(1955), (2)1 for Chabrier(2003), and (3)2 for Kroupa(2001).

	:param duste_switch: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the dust emission modeling.

	:param add_neb_emission: (default: 1)
		Choice for turning on (1) or off (0) the nebular emission modeling.

	:param dust_ext_law: (default: 'Cal2000')
		Choice for the dust attenuation law. Options are: (1)'CF2000' or 0 for Charlot & Fall (2000), (2)'Cal2000' or 1 for Calzetti et al. (2000).

	:param sfh_form: (default: 'delayed_tau_sfh')
		Choice for the parametric SFH model. Options are: (1)0 or 'tau_sfh', (2)1 or 'delayed_tau_sfh', (3)2 or 'log_normal_sfh', (4)3 or 'gaussian_sfh', (5)4 or 'double_power_sfh'.

	:param add_agn: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the AGN dusty torus modeling.

	:param add_igm_absorption: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the IGM absorption modeling.

	:param igm_type: (default: 'madau1995')
		Choice for the IGM absorption model. Only relevant if add_igm_absorption=1. Options are: (1)0 or 'madau1995' for Madau (1995), and (2)1 or 'inoue2014' for Inoue et al. (2014).

	:param cosmo: (default: 'flat_LCDM')
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param param_val:
		Dictionary of the input values of the parameters. Should folllow the structure given in the default set. Available free parameters are the same as those tabulated in Table 1 of `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.

	:returns spec_SED:
		Arrays containing output model spectrum. It consists of spec_SED['wave'] which is the wavelengths grid and spec_SED['flux'] which is the fluxes or the spectrum. 
	"""
	
	# input parameters:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	age = math.pow(10.0,params_val['log_age'])
	tau = math.pow(10.0,params_val['log_tau'])
	t0 = math.pow(10.0,params_val['log_t0'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	if sp == None:
		sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type)

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
		sp.params['gas_logu'] = gas_logu
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False

	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,params_val['log_fagn'])
		sp.params["agn_tau"] = math.pow(10.0,params_val['log_tauagn'])
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# dust atenuation:
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

	# generate the SED:
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form==0 or sfh_form==1:
		if sfh_form=='tau_sfh' or sfh_form==0:
			sp.params["sfh"] = 1
		elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
			sp.params["sfh"] = 4
		sp.params["const"] = 0
		sp.params["sf_start"] = 0
		sp.params["sf_trunc"] = 0
		sp.params["fburst"] = 0
		sp.params["tburst"] = 30.0
		sp.params["tau"] = tau
		#sp.params["tage"] = age

		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		# get model mass:
		mass = sp.stellar_mass
		# get dust mass: 
		#dust_mass0 = sp.dust_mass    # in solar mass/norm

		# normalization:
		norm0 = formed_mass/mass
		extnc_spec = extnc_spec*norm0

	elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==2 or sfh_form==3 or sfh_form==4:
		sp.params["sfh"] = 3
		SFR_fSM,mass,wave,extnc_spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
		
	# redshifting
	#redsh_wave,redsh_spec0 = redshifting.cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)
	redsh_wave,redsh_spec0 = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	redsh_spec = redsh_spec0
	if add_igm_absorption == 1:
		if igm_type==0 or igm_type=='madau1995':
			trans = igm_att_madau(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans
		elif igm_type==1 or igm_type=='inoue2014':
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans

	# normalize:
	#norm0 = formed_mass/mass
	#redsh_spec = redsh_spec0*norm0

	spec_SED = {}
	spec_SED['wave'] = redsh_wave
	spec_SED['flux'] = redsh_spec

	return spec_SED


def generate_modelSED_photo(sp=None,imf_type=1,duste_switch=0,add_neb_emission=1,dust_ext_law='Cal2000',sfh_form='delayed_tau_sfh',
	add_agn=0,filters=[],add_igm_absorption=0,igm_type='madau1995',cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0,params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):
	"""Function for generating a model photometric SED given some parameters.

	:param sp: (optional, default: None)
		Initialization of FSPS, such as `sp=fsps.StellarPopulation()`.

	:param imf_type: (default: 1)
		Choice for the IMF. Choices are: (1)0 for Salpeter(1955), (2)1 for Chabrier(2003), and (3)2 for Kroupa(2001).

	:param duste_switch: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the dust emission modeling.

	:param add_neb_emission: (default: 1)
		Choice for turning on (1) or off (0) the nebular emission modeling.

	:param dust_ext_law: (default: 'Cal2000')
		Choice for the dust attenuation law. Options are: (1)'CF2000' or 0 for Charlot & Fall (2000), (2)'Cal2000' or 1 for Calzetti et al. (2000).

	:param sfh_form: (default: 'delayed_tau_sfh')
		Choice for the parametric SFH model. Options are: (1)0 or 'tau_sfh', (2)1 or 'delayed_tau_sfh', (3)2 or 'log_normal_sfh', (4)3 or 'gaussian_sfh', (5)4 or 'double_power_sfh'.

	:param add_agn: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the AGN dusty torus modeling.

	:param filters:
		List of the photometric filters. This is mandatory parameter, in other word it should not empty. The accepted naming for the filters can be seen using :func:`list_filters` function in the :mod:`utils.filtering` module.

	:param add_igm_absorption: (default: 0)
		Choice for turning on (value: 1) or off (value: 0) the IGM absorption modeling.

	:param igm_type: (default: 'madau1995')
		Choice for the IGM absorption model. Only relevant if add_igm_absorption=1. Options are: (1)0 or 'madau1995' for Madau (1995), and (2)1 or 'inoue2014' for Inoue et al. (2014).

	:param cosmo: (default: 'flat_LCDM')
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param param_val:
		Dictionary of the input values of the parameters. Should folllow the structure given in the default set. Available free parameters are the same as those tabulated in Table 1 of `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.

	:returns photo_SED:
		Arrays containing output model photometric SED. It consists of photo_SED['wave'] which is the central wavelengths of the photometric filters and photo_SED['flux'] which is the photometric fluxes. 
	"""

	# Input parameters
	formed_mass = math.pow(10.0,params_val['log_mass'])
	age = math.pow(10.0,params_val['log_age'])
	tau = math.pow(10.0,params_val['log_tau'])
	t0 = math.pow(10.0,params_val['log_t0'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	# Initialize FSPS
	if sp == None:
		sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type)

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
		sp.params['gas_logu'] = gas_logu
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False

	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,params_val['log_fagn'])
		sp.params["agn_tau"] = math.pow(10.0,params_val['log_tauagn'])
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# dust atenuation:
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

	# generate the SED:
	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form==0 or sfh_form==1:
		if sfh_form=='tau_sfh' or sfh_form==0:
			sp.params["sfh"] = 1
		elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
			sp.params["sfh"] = 4
		sp.params["const"] = 0
		sp.params["sf_start"] = 0
		sp.params["sf_trunc"] = 0
		sp.params["fburst"] = 0
		sp.params["tburst"] = 30.0
		sp.params["tau"] = tau
		#sp.params["tage"] = age

		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		# get model mass:
		mass = sp.stellar_mass
		# get dust mass: 
		#dust_mass0 = sp.dust_mass   ## in solar mass/norm

		# normalization:
		norm0 = formed_mass/mass
		extnc_spec = extnc_spec*norm0

	elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==2 or sfh_form==3 or sfh_form==4:
		sp.params["sfh"] = 3
		SFR_fSM,mass,wave,extnc_spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
	else:
		print ("SFH choice is not recognized!")
		sys.exit()
		
	# redshifting
	#redsh_wave,redsh_spec0 = redshifting.cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)
	redsh_wave,redsh_spec0 = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	redsh_spec = redsh_spec0
	if add_igm_absorption == 1:
		if igm_type==0 or igm_type=='madau1995':
			trans = igm_att_madau(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans
		elif igm_type==1 or igm_type=='inoue2014':
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans

	# normalize:
	#norm0 = formed_mass/mass
	#redsh_spec = redsh_spec0*norm0
	#dust_mass = dust_mass0*norm0

	# filtering:
	photo_SED_flux = filtering(redsh_wave,redsh_spec,filters)
	# get central wavelength of all filters:
	photo_cwave = cwave_filters(filters)

	photo_SED = {}
	photo_SED['wave'] = photo_cwave
	photo_SED['flux'] = photo_SED_flux

	return photo_SED

def generate_modelSED_specphoto(sp=None,imf_type=1,duste_switch=1,add_neb_emission=1,dust_ext_law='Cal2000',sfh_form='delayed_tau_sfh',
	add_agn=0,filters=['galex_fuv','galex_nuv','sdss_u','sdss_g','sdss_r','sdss_i','sdss_z'],add_igm_absorption=0,igm_type='inoue2014',
	cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0,params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,
	'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
	'log_tau':0.4,'logzsol':0.0}):
	"""A function to generate model spectrophotometric SED

	:param sp:
		Initialization of FSPS, such as sp=fsps.StellarPopulation()

	:param imf_type:
		Choice for the IMF. Choices are: [0:Salpeter(1955), 1:Chabrier(2003), 2:Kroupa(2001)]

	:param duste_switch:
		Choice for turning on (1) or off (0) the dust emission modeling

	:param add_neb_emission:
		Choice for turning on (1) or off (0) the nebular emission modeling

	:param dust_ext_law:
		Choice of dust attenuation law. Options are: ['CF2000':Charlot&Fall(2000), 'Cal2000':Calzetti+2000]

	:param sfh_form:
		Choice for the parametric SFH model. 
		Options are: ['tau_sfh', 'delayed_tau_sfh', 'log_normal_sfh', 'gaussian_sfh', 'double_power_sfh']

	:param add_agn:
		Choice for turning on (1) or off (0) the AGN dusty torus modeling

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

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param param_val:
		A dictionary of parameters values.
	"""
	
	# some input parameters:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	age = math.pow(10.0,params_val['log_age'])
	tau = math.pow(10.0,params_val['log_tau'])
	t0 = math.pow(10.0,params_val['log_t0'])
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
		sp.params['gas_logu'] = gas_logu
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False

	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,params_val['log_fagn'])
		sp.params["agn_tau"] = math.pow(10.0,params_val['log_tauagn'])
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# dust atenuation:
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

	if sfh_form=='tau_sfh' or sfh_form=='delayed_tau_sfh' or sfh_form==0 or sfh_form==1:
		if sfh_form=='tau_sfh' or sfh_form==0:
			sp.params["sfh"] = 1
		elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
			sp.params["sfh"] = 4
		sp.params["const"] = 0
		sp.params["sf_start"] = 0
		sp.params["sf_trunc"] = 0
		sp.params["fburst"] = 0
		sp.params["tburst"] = 30.0
		sp.params["tau"] = tau
		#sp.params["tage"] = age
		
		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		# get model mass:
		mass = sp.stellar_mass
		# get dust mass: 
		#dust_mass0 = sp.dust_mass   # in solar mass/norm

		# normalization:
		norm0 = formed_mass/mass
		extnc_spec = extnc_spec*norm0
	elif sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh' or sfh_form=='double_power_sfh' or sfh_form==2 or sfh_form==3 or sfh_form==4:
		sp.params["sfh"] = 3
		SFR_fSM,mass,wave,extnc_spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
		
	# redshifting
	#redsh_wave,redsh_spec0 = redshifting.cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)
	redsh_wave,redsh_spec0 = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	redsh_spec = redsh_spec0
	if add_igm_absorption == 1:
		if igm_type==0 or igm_type=='madau1995':
			trans = igm_att_madau(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans
		elif igm_type==1 or igm_type=='inoue2014':
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans

	# normalize:
	#norm0 = formed_mass/mass
	#redsh_spec = redsh_spec0*norm0
	#dust_mass = dust_mass0*norm0

	# filtering:
	photo_SED_flux = filtering(redsh_wave,redsh_spec,filters)
	# get central wavelength of all filters:
	photo_cwave = cwave_filters(filters)

	spec_SED = {}
	spec_SED['wave'] = redsh_wave
	spec_SED['flux'] = redsh_spec

	photo_SED = {}
	photo_SED['wave'] = photo_cwave
	photo_SED['flux'] = photo_SED_flux

	return spec_SED,photo_SED

def generate_modelSED_spec_decompose(sp=None,imf=1, duste_switch='duste',add_neb_emission=1,dust_ext_law='Cal2000',add_agn=1,
	add_igm_absorption=0,igm_type=1,sfh_form='delayed_tau_sfh',funit='erg/s/cm2/A',cosmo='flat_LCDM',H0=70.0,Om0=0.3,
	gas_logu=-2.0,params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,
	'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
	'log_tau':0.4,'logzsol':0.0}):
	"""A function for generating model spectroscopic SED and decompose the SED into its components.

	:param funit:
		Flux unit. Options are: [0/'erg/s/cm2/A', 1/'erg/s/cm2', 2/'Jy']
	
	"""

	# allocate memories:
	spec_SED = {}
	spec_SED['wave'] = []
	spec_SED['flux_total'] = []
	spec_SED['flux_stellar'] = []
	spec_SED['flux_nebe'] = []
	spec_SED['flux_duste'] = []
	spec_SED['flux_agn'] = []

	# generate model spectrum: total
	spec_SED_tot = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=duste_switch,add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,
					sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
	spec_SED['wave'] = spec_SED_tot['wave']
	spec_SED['flux_total'] = convert_unit_spec_from_ergscm2A(spec_SED_tot['wave'],spec_SED_tot['flux'],funit=funit)

	spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=0,add_neb_emission=0,dust_ext_law=dust_ext_law,sfh_form=sfh_form,add_agn=0,
					add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
	spec_flux_stellar = spec_SED_temp['flux']
	spec_SED['flux_stellar'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_stellar,funit=funit)

	# get nebular emission:
	if add_neb_emission == 1:
		add_neb_emission_temp  = 1
		spec_SED_temp1 = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch='noduste',add_neb_emission=add_neb_emission_temp,
							dust_ext_law=dust_ext_law,sfh_form=sfh_form,add_agn=0,add_igm_absorption=add_igm_absorption,igm_type=igm_type,
							cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		add_neb_emission_temp  = 0
		spec_SED_temp2 = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch='noduste',add_neb_emission=add_neb_emission_temp,
							dust_ext_law=dust_ext_law,sfh_form=sfh_form,add_agn=0,add_igm_absorption=add_igm_absorption,igm_type=igm_type,
							cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		spec_flux_nebe = spec_SED_temp1['flux'] - spec_SED_temp2['flux']
		spec_SED['flux_nebe'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_nebe,funit=funit)

	# get dust emission:
	if duste_switch == 'duste' or duste_switch == 1:
		duste_switch_temp = 0
		spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=duste_switch_temp,add_neb_emission=add_neb_emission,
							dust_ext_law=dust_ext_law,sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,
							cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		spec_flux_duste = spec_SED_tot['flux'] - spec_SED_temp['flux']
		spec_SED['flux_duste'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_duste,funit=funit)
	# get AGN emission:
	if add_agn == 1:
		add_agn_temp = 0
		spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=duste_switch,add_neb_emission=add_neb_emission,
							dust_ext_law=dust_ext_law,sfh_form=sfh_form,add_agn=add_agn_temp,add_igm_absorption=add_igm_absorption,
							igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		spec_flux_agn = spec_SED_tot['flux'] - spec_SED_temp['flux']
		spec_SED['flux_agn'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_agn,funit=funit)


	return spec_SED


def generate_modelSED_specphoto_decompose(sp=None, imf=1, duste_switch='duste',add_neb_emission=1,dust_ext_law='Cal2000',
	add_agn=1,add_igm_absorption=0,igm_type=1,cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0,sfh_form='delayed_tau_sfh',
	funit='erg/s/cm2/A',filters=['galex_fuv','galex_nuv','sdss_u','sdss_g','sdss_r','sdss_i','sdss_z'],
	params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
	'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,
	'logzsol':0.0}):
	"""A function for generating model spectroscopic SED and decompose the SED into its components.
	
	:param funit:
		Flux unit. Options are: [0/'erg/s/cm2/A', 1/'erg/s/cm2', 2/'Jy']

	"""

	# allocate memories:
	spec_SED = {}
	spec_SED['wave'] = []
	spec_SED['flux_total'] = []
	spec_SED['flux_stellar'] = []
	spec_SED['flux_nebe'] = []
	spec_SED['flux_duste'] = []
	spec_SED['flux_agn'] = []

	photo_SED = {}
	photo_SED['wave'] = cwave_filters(filters)
	photo_SED['flux'] = []

	# generate model spectrum: total
	spec_SED_tot,photo_SED_tot = generate_modelSED_specphoto(sp=sp,imf_type=imf,duste_switch=duste_switch,add_neb_emission=add_neb_emission,
										dust_ext_law=dust_ext_law,sfh_form=sfh_form,add_agn=add_agn,filters=filters,add_igm_absorption=add_igm_absorption,
										igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
	spec_SED['wave'] = spec_SED_tot['wave']
	spec_SED['flux_total'] = convert_unit_spec_from_ergscm2A(spec_SED_tot['wave'],spec_SED_tot['flux'],funit=funit)

	photo_SED['flux'] = convert_unit_spec_from_ergscm2A(photo_SED_tot['wave'],photo_SED_tot['flux'],funit=funit)

	spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=0,add_neb_emission=0,dust_ext_law=dust_ext_law,sfh_form=sfh_form,
											add_agn=0,add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,
											params_val=params_val)
	spec_flux_stellar = spec_SED_temp['flux']
	spec_SED['flux_stellar'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_stellar,funit=funit)


	# get nebular emission:
	if add_neb_emission == 1:
		spec_SED_temp1 = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch='noduste',add_neb_emission=1,dust_ext_law=dust_ext_law,
							sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,
							params_val=params_val)

		spec_SED_temp2 = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch='noduste',add_neb_emission=0,dust_ext_law=dust_ext_law,
							sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,
							params_val=params_val)

		spec_flux_nebe = spec_SED_temp1['flux'] - spec_SED_temp2['flux']
		spec_SED['flux_nebe'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_nebe,funit=funit)

	# get dust emission:
	if duste_switch == 'duste' or duste_switch == 1:
		duste_switch_temp = 0
		spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=duste_switch_temp,add_neb_emission=add_neb_emission,
							dust_ext_law=dust_ext_law,sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,
							cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		spec_flux_duste = spec_SED_tot['flux'] - spec_SED_temp['flux']
		spec_SED['flux_duste'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_duste,funit=funit)
	# get AGN emission:
	if add_agn == 1:
		add_agn_temp = 0
		spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=duste_switch,add_neb_emission=add_neb_emission,
							dust_ext_law=dust_ext_law,sfh_form=sfh_form,add_agn=add_agn_temp,add_igm_absorption=add_igm_absorption,igm_type=igm_type,
							cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		spec_flux_agn = spec_SED_tot['flux'] - spec_SED_temp['flux']
		spec_SED['flux_agn'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_agn,funit=funit)

	return spec_SED,photo_SED


def save_models(gal_z=None,filters=[],specphoto=0,spec_sigma=5.0,wave_range=[3000,10000],imf_type=1,sfh_form=4,dust_ext_law=1,
	add_igm_absorption=0,igm_type=1,duste_switch=0,add_neb_emission=1,add_agn=0,gas_logu=-2.0,npmod_seds=100000, logzsol_range=[-2.0,0.2],
	log_tau_range=[-1.0,1.5],log_age_range=[-1.0,1.14],log_alpha_range=[-2.0,2.0],log_beta_range=[-2.0,2.0],log_t0_range=[-1.0,1.14],
	dust_index_range=[-0.7,-0.7],dust1_range=[0.0,3.0],dust2_range=[0.0,3.0],log_gamma_range=[-3.0,-0.824],log_umin_range=[-1.0,1.176],
	log_qpah_range=[-1.0,0.845],z_range=[0,0], log_fagn_range=[-5.0,0.48],log_tauagn_range=[0.70,2.18],nproc=10,cosmo=0,H0=70.0,Om0=0.3,
	name_out_fits=None):
	"""Function for generating pre-calculated model SEDs and store them into a FITS file. This is supposed to be created one for one redshift (i.e, one galaxy or a group of galaxies with similar redshift). 
	This FITS file can be used in fitting all the spatial bins in the galaxy (or a group of galaxies with similar redshift).  

	:param gal_z: (Mandatory)
		Galaxy's redshift. This parameter is mandatory.

	:param filters: (default: [])
		Set of photometric filters. The accepted naming for the filters can be seen using :func:`list_filters` function in the :mod:`utils.filtering` module.

	:param specphoto: (default: 0)
		Flag stating whether to produce model spectrophotometric SEDs (value:1) or model photometric SEDs (value: 0).

	:param spec_sigma: (defult: 5)
		Spectral resolution in sigma (in unit of Angstrom). Only relevant if specphoto=1.

	:param wave_range: (default: [3000,10000])
		Wavelength range of the model spectra to be generated. Only relevant if specphoto=1.   

	:param imf_type: (default: 1)
		Choice for the IMF. Options are: (1)0 for Salpeter(1955), (2)1 for Chabrier(2003), and (3)2 for Kroupa(2001).

	:param sfh_form: (default: 1)
		Choice for the SFH model. Options are: (1)0 or 'tau_sfh', (2)1 or 'delayed_tau_sfh', (3)2 or 'log_normal_sfh', (4)3 or 'gaussian_sfh', and (5)4 or 'double_power_sfh'.

	:param dust_ext_law: (default: 1)
		Choice for the dust attenuation law. Options are: (1)'CF2000' or 0 for Charlot & Fall (2000), (2)'Cal2000' or 1 for Calzetti et al. (2000).

	:param add_igm_absorption: (default: 0)
		Switch for the IGM absorption. Options are: (1)0 means turn off, and (2)1 means turn on.

	:param igm_type: (default: 1)
		Choice for the IGM absorption model. Options are: (1)0 or 'madau1995' for Madau (1995), and (2)1 or 'inoue2014' for Inoue et al. (2014).

	:param duste_switch: (default: 0)
		Switch for the dust emission modeling. Options are: (1)0 means turn off, and (2)1 means turn on.

	:param add_neb_emission: (default: 1)
		Switch for the nebular emission modeling. Options are: (1)0 means turn off, and (2)1 means turn on.

	:param add_agn: (default: 0)
		Switch for the AGN dusty torus emission modeling. Options are: (1)0 means turn off, and (2)1 means turn on.

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in log scale.

	:param npmod_seds: (default: 100000)
		Number of model SEDs to be generated.

	:param nproc: (default: 10)
		Number of processors (cores) to be used in the calculation.

	:param cosmo: (default: 0)
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param name_out_fits: (optional, default:None)
		Desired name for the output FITS file.

	:returns name_out_fits:
		Output FITS file.
	"""

	# get number of filters:
	nbands = len(filters)

	# make text file containing list of filters:
	name_filters_list = "filters_list%d.dat" % (random.randint(0,10000))
	file_out = open(name_filters_list,"w")
	for ii in range(0,nbands):
		file_out.write("%s\n" % filters[int(ii)]) 
	file_out.close()

	# store the text into the temp directory:
	dir_file = PIXEDFIT_HOME+'/data/temp/'
	os.system('mv %s %s' % (name_filters_list,dir_file))

	# make configuration file:
	name_config = "config_file%d.dat" % (random.randint(0,10000))
	file_out = open(name_config,"w")
	if specphoto == 1:
		file_out.write("spec_sigma %lf\n" % spec_sigma)
		file_out.write("min_wave %lf\n" % wave_range[0])
		file_out.write("max_wave %lf\n" % wave_range[1])
	file_out.write("imf_type %d\n" % imf_type)
	file_out.write("add_neb_emission %d\n" % add_neb_emission)
	file_out.write("gas_logu %lf\n" % gas_logu)
	file_out.write("add_igm_absorption %d\n" % add_igm_absorption)

	# SFH choice
	if sfh_form=='tau_sfh' or sfh_form==0:
		sfh_form1 = 0
	elif sfh_form=='delayed_tau_sfh' or sfh_form==1:
		sfh_form1 = 1
	elif sfh_form=='log_normal_sfh' or sfh_form==2:
		sfh_form1 = 2
	elif sfh_form=='gaussian_sfh' or sfh_form==3:
		sfh_form1 = 3
	elif sfh_form=='double_power_sfh' or sfh_form==4:
		sfh_form1 = 4
	else:
		print ("SFH choice is not recognized!")
		sys.exit()
	file_out.write("sfh_form %d\n" % sfh_form1)

	# dust law
	if dust_ext_law=='CF2000' or dust_ext_law==0:
		dust_ext_law1 = 0
	elif dust_ext_law=='Cal2000' or dust_ext_law==1:
		dust_ext_law1 = 1
	else:
		print ("dust_ext_law is not recognized!")
		sys.exit()
	file_out.write("dust_ext_law %d\n" % dust_ext_law1)

	# IGM type
	if igm_type=='madau1995' or igm_type==0:
		igm_type1 = 0
	elif igm_type=='inoue2014' or igm_type==1:
		igm_type1 = 1
	else:
		print ("igm_type is not recognized!")
		sys.exit()
	file_out.write("igm_type %d\n" % igm_type1)

	file_out.write("duste_switch %d\n" % duste_switch)
	file_out.write("add_agn %d\n" % add_agn)
	file_out.write("npmod_seds %d\n" % npmod_seds)
	file_out.write("pr_logzsol_min %lf\n" % logzsol_range[0])
	file_out.write("pr_logzsol_max %lf\n" % logzsol_range[1])
	file_out.write("pr_log_tau_min %lf\n" % log_tau_range[0])
	file_out.write("pr_log_tau_max %lf\n" % log_tau_range[1])
	file_out.write("pr_log_t0_min %lf\n" % log_t0_range[0])
	file_out.write("pr_log_t0_max %lf\n" % log_t0_range[1])
	file_out.write("pr_log_alpha_min %lf\n" % log_alpha_range[0])
	file_out.write("pr_log_alpha_max %lf\n" % log_alpha_range[1])
	file_out.write("pr_log_beta_min %lf\n" % log_beta_range[0])
	file_out.write("pr_log_beta_max %lf\n" % log_beta_range[1])
	file_out.write("pr_log_age_min %lf\n" % log_age_range[0])
	file_out.write("pr_log_age_max %lf\n" % log_age_range[1])
	file_out.write("pr_dust_index_min %lf\n" % dust_index_range[0])
	file_out.write("pr_dust_index_max %lf\n" % dust_index_range[1])
	file_out.write("pr_dust1_min %lf\n" % dust1_range[0])
	file_out.write("pr_dust1_max %lf\n" % dust1_range[1])
	file_out.write("pr_dust2_min %lf\n" % dust2_range[0])
	file_out.write("pr_dust2_max %lf\n" % dust2_range[1])
	file_out.write("pr_log_gamma_min %lf\n" % log_gamma_range[0])
	file_out.write("pr_log_gamma_max %lf\n" % log_gamma_range[1])
	file_out.write("pr_log_umin_min %lf\n" % log_umin_range[0])
	file_out.write("pr_log_umin_max %lf\n" % log_umin_range[1])
	file_out.write("pr_log_qpah_min %lf\n" % log_qpah_range[0])
	file_out.write("pr_log_qpah_max %lf\n" % log_qpah_range[1])
	file_out.write("pr_log_fagn_min %lf\n" % log_fagn_range[0])
	file_out.write("pr_log_fagn_max %lf\n" % log_fagn_range[1])
	file_out.write("pr_log_tauagn_min %lf\n" % log_tauagn_range[0])
	file_out.write("pr_log_tauagn_max %lf\n" % log_tauagn_range[1])

	# cosmology
	if cosmo=='flat_LCDM' or cosmo==0:
		cosmo1 = 0
	elif cosmo=='WMAP5' or cosmo==1:
		cosmo1 = 1
	elif cosmo=='WMAP7' or cosmo==2:
		cosmo1 = 2
	elif cosmo=='WMAP9' or cosmo==3:
		cosmo1 = 3
	elif cosmo=='Planck13' or cosmo==4:
		cosmo1 = 4
	elif cosmo=='Planck15' or cosmo==5:
		cosmo1 = 5
	#elif cosmo=='Planck18' or cosmo==6:
	#	cosmo1 = 6
	else:
		print ("Input cosmo is not recognized!")
		sys.exit()
	file_out.write("cosmo %d\n" % cosmo1)

	file_out.write("H0 %lf\n" % H0)
	file_out.write("Om0 %lf\n" % Om0)
	# output files name:
	if name_out_fits == None:
		name_out_fits = "random_modelSEDs.fits"
	file_out.write("name_out_fits %s\n" % name_out_fits)
	# galaxy's redshift:
	file_out.write("gal_z %lf\n" % gal_z)  
	file_out.close()
	# store configuration file into temp directory:
	dir_file = PIXEDFIT_HOME+'/data/temp/'
	os.system('mv %s %s' % (name_config,dir_file))

	# call the fitting functions:
	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_model/'
	if specphoto == 0:
		os.system("mpirun -n %d python %s./save_models_photo.py %s %s" % (nproc,CODE_dir,name_filters_list,name_config))
	elif specphoto == 1:
		os.system("mpirun -n %d python %s./save_models_specphoto.py %s %s" % (nproc,CODE_dir,name_filters_list,name_config))

	# free disk space:
	dir_file = PIXEDFIT_HOME+'/data/temp/'
	os.system("rm %s%s" % (dir_file,name_config))
	os.system("rm %s%s" % (dir_file,name_filters_list))

	return name_out_fits


### defne function to generate pre-calculated model SEDs with various values of redshifts
### each output fits file associated with one redshift
### at each redshift grid, random models have the same distribution of parameters
def save_models_gridz_params(npmod_seds=1000000, logzsol_range=[-2.0,0.2],logtau_range=[-1.0,1.5],logage_range=[-1.0,1.14],
	log_alpha_range=[-2.0,2.0],log_beta_range=[-2.0,2.0],logt0_range=[-1.0,1.14],dust_index_range=[-0.7,-0.7],dust1_range=[0.0,3.0],
	dust2_range=[0.0,3.0],loggamma_range=[-3.0,-0.824],logumin_range=[-1.0,1.176],logqpah_range=[-1.0,0.845],logfagn_range=[-5.0,0.48],
	logtauagn_range=[0.70,2.18],name_out_fits=None):
	"""A function for generating pre-calculated model SEDs with various values of redshifts. 
	Each output FITS file associated with one redshift. At each redshift grid, 
	random models have the same distribution of parameters.

	"""
	
	def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta','log_age','dust_index','dust1','dust2','log_gamma',
				'log_umin', 'log_qpah', 'log_fagn','log_tauagn']
	nparams_def = len(def_params) 

	# generate random parameters:
	random_params = {}
	random_params['logzsol'] = np.random.uniform(logzsol_range[0],logzsol_range[1],npmod_seds)
	random_params['log_tau'] = np.random.uniform(logtau_range[0],logtau_range[1],npmod_seds)
	random_params['log_t0'] = np.random.uniform(logt0_range[0],logt0_range[1],npmod_seds)
	random_params['log_alpha'] = np.random.uniform(log_alpha_range[0],log_alpha_range[1],npmod_seds)
	random_params['log_beta'] = np.random.uniform(log_beta_range[0],log_beta_range[1],npmod_seds)
	random_params['log_age'] = np.random.uniform(logage_range[0],logage_range[1],npmod_seds)
	if dust_index_range[0] != dust_index_range[1]:
		random_params['dust_index'] = np.random.uniform(dust_index_range[0],dust_index_range[1],npmod_seds)
	elif dust_index_range[0] == dust_index_range[1]:
		random_params['dust_index'] = np.zeros(npmod_seds) + dust_index_range[0]
	random_params['dust1'] = np.random.uniform(dust1_range[0],dust1_range[1],npmod_seds)
	random_params['dust2'] = np.random.uniform(dust2_range[0],dust2_range[1],npmod_seds)
	random_params['log_gamma'] = np.random.uniform(loggamma_range[0],loggamma_range[1],npmod_seds)
	random_params['log_umin'] = np.random.uniform(logumin_range[0],logumin_range[1],npmod_seds)
	random_params['log_qpah'] = np.random.uniform(logqpah_range[0],logqpah_range[1],npmod_seds)
	random_params['log_fagn'] = np.random.uniform(logfagn_range[0],logfagn_range[1],npmod_seds)
	random_params['log_tauagn'] = np.random.uniform(logtauagn_range[0],logtauagn_range[1],npmod_seds)

	# make fits file:
	hdul = fits.HDUList()

	# header:
	hdr = fits.Header()
	hdr['nparams_def'] = nparams_def
	for pp in range(0,nparams_def):
		str_temp = 'par%d' % pp
		hdr[str_temp] = def_params[pp] 
	hdr['pr_logzsol_min'] = logzsol_range[0]
	hdr['pr_logzsol_max'] = logzsol_range[1]
	hdr['pr_log_tau_min'] = logtau_range[0]
	hdr['pr_log_tau_max'] = logtau_range[1]
	hdr['pr_log_t0_min'] = logt0_range[0]
	hdr['pr_log_t0_max'] = logt0_range[1]
	hdr['pr_log_alpha_min'] = log_alpha_range[0]
	hdr['pr_log_alpha_max'] = log_alpha_range[1]
	hdr['pr_log_beta_min'] = log_beta_range[0]
	hdr['pr_log_beta_max'] = log_beta_range[1]
	hdr['pr_log_age_min'] = logage_range[0]
	hdr['pr_log_age_max'] = logage_range[1]
	hdr['pr_dust_index_min'] = dust_index_range[0]
	hdr['pr_dust_index_max'] = dust_index_range[1]
	hdr['pr_dust1_min'] = dust1_range[0]
	hdr['pr_dust1_max'] = dust1_range[1]
	hdr['pr_dust2_min'] = dust2_range[0]
	hdr['pr_dust2_max'] = dust2_range[1]
	hdr['pr_log_gamma_min'] = loggamma_range[0]
	hdr['pr_log_gamma_max'] = loggamma_range[1]
	hdr['pr_log_umin_min'] = logumin_range[0]
	hdr['pr_log_umin_max'] = logumin_range[1]
	hdr['pr_log_qpah_min'] = logqpah_range[0]
	hdr['pr_log_qpah_max'] = logqpah_range[1]
	hdr['pr_log_fagn_min'] = logfagn_range[0]
	hdr['pr_log_fagn_max'] = logfagn_range[1]
	hdr['pr_log_tauagn_min'] = logtauagn_range[0]
	hdr['pr_log_tauagn_max'] = logtauagn_range[1]
	hdr['npmod_seds'] = npmod_seds
	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul.append(primary_hdu)
	# parameters:
	cols0 = []
	ids = np.linspace(1,npmod_seds,npmod_seds)
	col = fits.Column(name='id', format='K', array=np.array(ids))
	cols0.append(col)
	for pp in range(0,nparams_def):
		col = fits.Column(name=def_params[pp], format='D', array=np.array(random_params[def_params[pp]]))
		cols0.append(col)
	cols = fits.ColDefs(cols0)
	hdul.append(fits.BinTableHDU.from_columns(cols, name='parameters'))

	if name_out_fits == None:
		name_out_fits = 'rand_params.fits'
	hdul.writeto(name_out_fits, overwrite=True)

	return name_out_fits


def save_models_gridz_sed(grid_z=[],name_fits_gridz=[],filters=[],name_params_fits='rand_params.fits',imf_type=1,
	sfh_form=1,dust_ext_law=1,add_igm_absorption=0,igm_type=1,duste_switch=1,add_neb_emission=1,add_agn=1,gas_logu=-2.0,
	nproc=10):

 	# get number of filters:
	nbands = len(filters)
	
	# get number of random model SEDs and whether dust_index is set fix or not 
	hdu = fits.open(name_params_fits)
	npmod_seds = int(hdu[0].header['npmod_seds']) 
	if hdu[1].data['dust_index'][0] == hdu[1].data['dust_index'][1]:
		fix_dust_index = 1
		fix_dust_index_val = hdu[1].data['dust_index'][0]
	else:
		fix_dust_index = 0
		fix_dust_index_val = 0
	hdu.close()


	ngrids_z = len(grid_z)
	for ii in range(0,ngrids_z):
		gal_z = grid_z[ii]
		name_out_fits = name_fits_gridz[ii]

		# make text file containing list of filters:
		name_filters_list = "filters_list%d.dat" % (random.randint(0,10000))
		file_out = open(name_filters_list,"w")
		for ii in range(0,nbands):
			file_out.write("%s\n" % filters[int(ii)]) 
		file_out.close()
		# store the text into the temp directory:
		dir_file = PIXEDFIT_HOME+'/data/temp/'
		os.system('mv %s %s' % (name_filters_list,dir_file))

		# (1) make configuration file:
		name_config = "config_file%d.dat" % (random.randint(0,10000))
		file_out = open(name_config,"w")
		file_out.write("imf_type %d\n" % imf_type)
		file_out.write("sfh_form %d\n" % sfh_form)
		file_out.write("add_neb_emission %d\n" % add_neb_emission)
		file_out.write("gas_logu %lf\n" % gas_logu)
		file_out.write("dust_ext_law %d\n" % dust_ext_law)
		file_out.write("add_igm_absorption %d\n" % add_igm_absorption)
		file_out.write("igm_type %d\n" % igm_type)
		file_out.write("duste_switch %d\n" % duste_switch)
		file_out.write("add_agn %d\n" % add_agn)
		file_out.write("npmod_seds %d\n" % npmod_seds)
		file_out.write("name_params_fits %s\n" % name_params_fits)
		file_out.write("fix_dust_index %d\n" % fix_dust_index)
		file_out.write("fix_dust_index_val %lf\n" % fix_dust_index_val)
		# output files name:
		if name_out_fits == None:
			name_out_fits = "random_modelSEDs.fits"
		file_out.write("name_out_fits %s\n" % name_out_fits)
		# galaxy's redshift:
		file_out.write("gal_z %lf\n" % gal_z)  
		file_out.close()
		# (2) store configuration file into temp directory:
		dir_file = PIXEDFIT_HOME+'/data/temp/'
		os.system('mv %s %s' % (name_config,dir_file))

		# (3) call the fitting functions:
		CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_model/'
		os.system("mpirun -n %d python %s./save_models_gridz.py %s %s" % (nproc,CODE_dir,name_filters_list,name_config))

		# (4) free disk space:
		dir_file = PIXEDFIT_HOME+'/data/temp/'
		os.system("rm %s%s" % (dir_file,name_config))
		os.system("rm %s%s" % (dir_file,name_filters_list))


def save_models_gridz_merge(list_name_fits=None,name_out_fits=None):
	"""A function to merge many FITS files into one FITS file. All files should have the same set of filters, 
	parameters, and number of models

	"""

	nfiles = len(list_name_fits)

	# get characterictics:
	hdu = fits.open(list_name_fits[0])
	header = hdu[0].header
	data_samplers = hdu[1].data
	hdu.close()

	# make the fits file:
	hdul = fits.HDUList()

	# header:
	hdr = fits.Header()

	hdr['imf_type'] = header['imf_type']
	hdr['sfh_form'] = header['sfh_form']
	hdr['dust_ext_law'] = header['dust_ext_law']
	hdr['duste_switch'] = header['duste_switch']
	hdr['add_neb_emission'] = header['add_neb_emission']
	hdr['gas_logu'] = header['gas_logu']
	hdr['add_agn'] = header['add_agn']
	hdr['add_igm_absorption'] = header['add_igm_absorption']
	if int(hdr['add_igm_absorption']) == 1:
		hdr['igm_type'] = header['igm_type']
	nfilters = int(header['nfilters'])
	hdr['nfilters'] = header['nfilters']

	for ii in range(0,nfilters):
		str_temp = 'fil%d' % ii
		hdr[str_temp] = header[str_temp]

	hdr['nrows'] = header['nrows']
	hdr['nparams'] = header['nparams']
	nparams = int(header['nparams'])
	params = []
	for ii in range(0,nparams):
		str_temp = 'param%d' % ii
		hdr[str_temp] = header[str_temp]
		params.append(header[str_temp])
	# get ranges of the parameters:
	for ii in range(0,nparams):
		str_temp = 'pr_%s_min' % params[ii]
		hdr[str_temp] = header[str_temp]
		str_temp = 'pr_%s_max' % params[ii]
		hdr[str_temp] = header[str_temp]

	# get list of redshifts:
	hdr['nz'] = nfiles
	for ii in range(0,nfiles):
		hdu = fits.open(list_name_fits[ii])
		str_temp = 'z%d' % ii
		hdr[str_temp] = float(hdu[0].header['gal_z'])
		hdu.close()

	primary_hdu = fits.PrimaryHDU(header=hdr)
	hdul.append(primary_hdu)

	# get the data:
	for ii in range(0,nfiles):
		hdu = fits.open(list_name_fits[ii])
		header = hdu[1].header
		data_samplers = hdu[1].data
		hdu.close()

		tfields = int(header['TFIELDS'])
		cols0 = []
		for jj in range(0,tfields):
			str_temp = 'TTYPE%d' % (jj+1)
			ttype = header[str_temp]
			str_temp = 'TFORM%d' % (jj+1)
			tform = header[str_temp]
			col = fits.Column(name=ttype, format=tform, array=np.array(data_samplers[ttype]))
			cols0.append(col)

		cols = fits.ColDefs(cols0)
		str_temp = 'z%d' % ii
		hdul.append(fits.BinTableHDU.from_columns(cols, name=str_temp))

		# end of for ii: nfiles
	if name_out_fits == None:
		name_out_fits = 'merged_savedmod_vz.fits'
	hdul.writeto(name_out_fits, overwrite=True)



def generate_mockSED(sp=None, imf_type=1,filters=['galex_fuv', 'galex_nuv', 'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z'],duste_switch=1,
					add_neb_emission=1,dust_ext_law='Cal2000',add_agn=1,add_igm_absorption=0,igm_type=1,sfh_form='delayed_tau_sfh',
					funit='erg/s/cm2/A',cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0, decompose=0, SNR=[], add_randnoise=1, 
					noise_treat=1,params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,
					'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,
					'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):
	"""Function to generate a simple mock SED (photometry and spectroscopy).

	:param sp:
		Initialization of FSPS, such as sp=fsps.StellarPopulation()

	:param imf_type:
		Choice for the IMF. Choices are: [0:Salpeter(1955), 1:Chabrier(2003), 2:Kroupa(2001)]

	:param filters:
		List of photometric filters.

	:param duste_switch:
		Choice for turning on (1) or off (0) the dust emission modeling

	:param add_neb_emission:
		Choice for turning on (1) or off (0) the nebular emission modeling

	:param dust_ext_law: (default: 1)
		Choice for the dust attenuation law. Options are: (1)'CF2000' or 0 for Charlot & Fall (2000), (2)'Cal2000' or 1 for Calzetti et al. (2000).

	:param add_agn:
		Choice for turning on (1) or off (0) the AGN dusty torus modeling

	:param add_igm_absorption: (default: 0)
		Switch for the IGM absorption.

	:param igm_type: (default: 1)
		Choice for the IGM absorption model. Options are: [0/'madau1995':Madau(1995); 1/'inoue2014':Inoue+(2014)]

	:param sfh_form: (default: 'delayed_tau_sfh')
		Choice for the parametric SFH model. 
		Options are: ['tau_sfh', 'delayed_tau_sfh', 'log_normal_sfh', 'gaussian_sfh', 'double_power_sfh', 'arbitrary_sfh']

	:param funit: (default: 'erg/s/cm2/A')
		Flux unit. Options are: (1)0 or 'erg/s/cm2/A', (2)1 or 'erg/s/cm2', and (3)2 or 'Jy'.

	:param cosmo: (default: 'flat_LCDM')
		Choices for the cosmological parameters. The choices are: ['flat_LCDM', 'WMAP5', 
		'WMAP7', 'WMAP9', 'Planck13', 'Planck15'], similar to the choices available in the 
		Astropy Cosmology package: https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies.
		If 'flat_LCDM' is chosen, the input H0 and Om0 should be provided.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param param_val:
		A dictionary of parameters values.

	:param decompose: (default: 1)
		Flag stating whether the mock spectroscopic SED is broken-down into its components (value: 1 or True) or not (value: 0 or False).

	:param SNR:
		Signal to noise ratios of the mock photometric SEDs is 1D array containing S/N in each filter. It should has the same length as filters.

	:param add_randnoise: (default: 1)
		Flag to add random Gaussian noises in the photometric fluxes.

	:param noise_treat: (default: 1)
		Choice in generating random Gaussian noise. Options are: 0 for generating one value from normal distribution, and 1 for generating 100 values from normal distribution then take one from them randomly.

	
	:returns mock_params:
		Parameters of mock SED.

	:returns mock_photo_SED:
		Mock photometric SED.

	:returns mock_spec_SED:
		Mock spectroscopic SED.

	"""

	# get number of filters:
	nbands = len(filters)
	if len(SNR) != nbands:
		print ("Number of elements in SNR must be the same as that of filters")
		sys.exit()

	# call fsps
	if sp == None:
		sp = fsps.StellarPopulation(zcontinuous=1)

	sp.params['imf_type'] = imf_type

	# generate model SED: total
	SED_prop,photo_SED,spec_SED = generate_modelSED_propspecphoto(sp=sp,duste_switch=duste_switch,add_neb_emission=add_neb_emission,
													dust_ext_law=dust_ext_law,sfh_form=sfh_form,add_agn=add_agn,filters=filters,
													add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,
													gas_logu=gas_logu,params_val=params_val)
	tot_wave = spec_SED['wave']
	tot_spec = spec_SED['flux'] 
	#[0/'erg/s/cm2/A', 1/'erg/s/cm2', 2/'Jy']
	if funit=='erg/s/cm2/A' or funit==0:
		tot_spec = tot_spec
	elif funit=='erg/s/cm2' or funit==1:
		tot_spec = np.asarray(tot_spec)*np.asarray(tot_wave)
	elif funit=='Jy' or funit==2:
		tot_spec = np.asarray(tot_spec)*np.asarray(tot_wave)*np.asarray(tot_wave)/1.0e-23/2.998e+18
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	mock_params = {}
	mock_params['SM'] = SED_prop['SM']
	#mock_params['survive_mass'] = SED_prop['survive_mass']
	mock_params['SFR'] = SED_prop['SFR']
	mock_params['mw_age'] = SED_prop['mw_age'] 
	mock_params['dust_mass'] = SED_prop['dust_mass']
	mock_params['log_fagn_bol'] = SED_prop['log_fagn_bol']

	if funit=='erg/s/cm2/A' or funit==0:
		photo_flux = photo_SED['flux']
	elif funit == 'erg/s/cm2' or funit==1:
		photo_flux = np.asarray(photo_SED['flux'])*np.asarray(photo_SED['wave'])
	elif funit == 'Jy' or funit==2:
		photo_flux = np.asarray(photo_SED['flux'])*np.asarray(photo_SED['wave'])*np.asarray(photo_SED['wave'])/1.0e-23/2.998e+18
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	mock_photo_SED = {}
	mock_photo_SED['wave'] = photo_SED['wave']

	# calculate flux errors: if add_randnoise=1
	if len(SNR) == 0:
		SNR = np.zeros(nbands) + 10.0

	mock_fluxes_err = np.zeros(nbands)
	for bb in range(0,nbands):
		mock_fluxes_err[bb] = photo_flux[bb]/SNR[bb]
	mock_photo_SED['flux_err'] = mock_fluxes_err
	# calculate photo. flux:
	if add_randnoise == 1:
		photo_flux1 = np.zeros(nbands)

		if noise_treat == 0:
			for bb in range(0,nbands):
				photo_flux1[bb] = np.random.normal(photo_flux[bb],mock_fluxes_err[bb],1)
		elif noise_treat == 1:
			for bb in range(0,nbands):
				rand_values = np.random.normal(photo_flux[bb],mock_fluxes_err[bb],100)
				rand_idx = random.randint(0,99)
				photo_flux1[bb] = rand_values[int(rand_idx)]

		mock_photo_SED['flux'] = photo_flux1		
	elif add_randnoise == 0:
		mock_photo_SED['flux'] = photo_flux

	mock_spec_SED = {}
	if decompose == 1:
		mock_spec_SED['tot'] = {}
		mock_spec_SED['tot']['wave'] = tot_wave
		mock_spec_SED['tot']['flux'] = tot_spec

		mock_spec_SED['stellar'] = {}
		mock_spec_SED['duste'] = {}
		mock_spec_SED['agn'] = {}
		mock_spec_SED['nebe'] = {}
	elif decompose == 0:
		mock_spec_SED['wave'] = tot_wave
		mock_spec_SED['flux'] = tot_spec

	if decompose == 1:
		spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf_type, duste_switch=duste_switch, add_neb_emission=add_neb_emission,
													dust_ext_law=dust_ext_law, add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,sfh_form=sfh_form,
													funit=funit,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu)

		mock_spec_SED['stellar']['wave'] = spec_SED['wave']
		mock_spec_SED['stellar']['flux'] = spec_SED['flux_stellar']

		mock_spec_SED['nebe']['wave'] = spec_SED['wave']
		mock_spec_SED['nebe']['flux'] = spec_SED['flux_nebe']

		mock_spec_SED['duste']['wave'] = spec_SED['wave']
		mock_spec_SED['duste']['flux'] = spec_SED['flux_duste']

		mock_spec_SED['agn']['wave'] = spec_SED['wave']
		mock_spec_SED['agn']['flux'] = spec_SED['flux_agn']
	
	return mock_params,mock_photo_SED,mock_spec_SED



def add_fagn_bol_samplers(name_sampler_fits=None, name_out_fits=None, nproc=10):
	"""Function to add f_agn_bol into the FITS file containing MCMC samplers. 
	This parameter means a fraction of the total bolometric luminosity that is come from AGN contribution.
	This is different to the f_agn (native of FSPS and one of free parameters in SED fitting) which means 
	the ratio between AGN bolometric luminosity and the stellar bolometric luminosity.

	:param name_sampler_fits: (default: None)
		Name of the input FITS file that contains the MCMC samplers.

	:param name_out_fits: (default: None)
		Desired name for the output FITS file. If None, the name will be 'add_fagnbol_[name_sampler_fits]'.

	:param nproc: (default: 10)
		Number of cores to be used.   
	"""
	if name_sampler_fits == None:
		print ("name_sampler_fits can't be None!...")
		sys.exit()
	else:
		if name_out_fits == None:
			name_out_fits = 'add_fagnbol_%s' % name_sampler_fits

	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_model/'
	os.system("mpirun -n %d python %s./add_fagnbol.py %s %s" % (nproc,CODE_dir,name_sampler_fits,name_out_fits))

	return name_out_fits



