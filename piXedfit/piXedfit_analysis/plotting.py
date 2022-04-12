import numpy as np
import math
import sys, os
import random
import fsps
import operator
import matplotlib
matplotlib.use('agg')
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import *

from ..piXedfit_model import generate_modelSED_spec_decompose, construct_SFH
from ..utils.filtering import cwave_filters, filtering
from ..utils.posteriors import plot_triangle_posteriors


__all__ = ["plot_SED_rdsps", "plot_SED_rdsps_with_residual", "plot_SED_mcmc", "plot_SED_mcmc_with_residual", 
			"plot_corner", "plot_sfh_mcmc", "plot_SED_rdsps_save_PDF"]



def plot_SED_rdsps(name_sampler_fits=None, logscale_x=True, logscale_y=True, xrange=None, yrange=None, 
	wunit='micron', funit='erg/s/cm2/A', decompose=1, xticks=None, photo_color='red', fontsize_tick=18,
	fontsize_label=25, fontsize_legend=18, markersize=100, lw=2.0, name_plot=None):

	"""Function for producing an SED plot from a fitting result obtained with the RDSPS method. 
	In this case, the best-fit model SED in the plot is the one with lowest chi-square from the input set of pre-calculated model SEDs in the fitting. 

	:param name_sampler_fits: (Mandatory, default: None)
		Name of input FITS file containing model SEDs and their probabilities. This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.

	:param logscale_x: (optional, default: True)
		Flag stating whether the x-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param logscale_y: (optional, default: True)
		Flag stating whether the y-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param xrange: (optional, default: None)
		Range in x-axis. The accepted format is: [xmin,xmax]. If xrange=None, the range will be defined based on 
		the wavelength range covered by the observed photometric SED.

	:param yrange: (optional, default: None)
		Range in y-axis. The accepted format is: [ymin,ymax]. If yrange=None, the range will be defined based on
		the fluxes range covered by the observed photometric SED.  
	
	:param wunit: (default: 'micron')
		Wavelength unit. Options are: (1)0 or 'angstrom' for Angstrom unit and (2)1 or 'micron' for micron unit.

	:param funit: (default: 'erg/s/cm2/A')
		Flux unit. Options are: (1)0 or 'erg/s/cm2/A', (2)1 or 'erg/s/cm2', and (3)2 or 'Jy'.

	:param decompose: (default: 1)
		Flag stating whether the best-fit model SED is broken-down into its components (value: 1 or True) or not (value: 0 or False).

	:param xticks: (optional, default: None)
		List of ticks values in x-axis. If None, the default from matplotlib is used. If xticks is not None, the accepted input is in list format  
		xticks = []. The unit should be the same as the input wunit.

	:param photo_color: (optional, default: 'red')
		Color of photometric fluxes points (in string). The accepted input is the same as that available in matplotlib.

	:param fontsize_tick: (optional, default: 20)
		Fontsize for the tick. Only relevant if xticks is not None. 
	
	:param fontsize_label: (optional, default: 25)
		Fontsize for the x- and y-axis labels. 

	:param fontsize_legend: (optional, default: 18)
		Fontsize for the legend.

	:param markersize: (optional, default: 100)
		Size of the maarkers associated with the observed and model SEDs.

	:param lw: (optional, default: 1)
		Line width of the model SEDs.
	
	:param name_plot: (optional, default: None)
		Name of the output plot. This is optional parameter.
	

	:returns name_plot:
		Name of the output plot.

	:returns spec_wave:
		Wavelength grids of the total best-fit model spectrum.

	:returns spec_total:
		Fluxes grids of the total best-fit model spectrum.

	:returns spec_stellar:
		Stellar emission component of the best-fit model spectrum.

	:returns spec_nebe:
		Nebular emission component of the best-fit model spectrum.

	:returns spec_duste:
		Dust emission component of the best-fit model spectrum.

	:returns spec_agn:
		AGN dusty torus emission component of the best-fit model spectrum.
	"""

	def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta', 'log_age','dust_index','dust1','dust2',
					'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

	def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

	# open the FITS file:
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	if header_samplers['col1'] == 'rows':
		store_full_samplers = 0
		data_bfit_spec = hdu[2].data
		data_bfit_photo = hdu[3].data
	elif header_samplers['col1'] == 'id':
		store_full_samplers = 1
		data_samplers = hdu[1].data
	hdu.close()

	# filters and observed SED
	nbands = int(header_samplers['nfilters'])
	filters = []
	obs_fluxes = np.zeros(nbands)
	obs_flux_err = np.zeros(nbands)
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		filters.append(header_samplers[str_temp])
		str_temp = 'flux%d' % bb
		obs_fluxes[bb] = float(header_samplers[str_temp])
		str_temp = 'flux_err%d' % bb
		obs_flux_err[bb] = float(header_samplers[str_temp])

	# central wavelength of all filters
	photo_cwave = cwave_filters(filters)

	
	if store_full_samplers == 1:
		# some parameters 
		imf = int(header_samplers['imf'])
		sfh_form = header_samplers['sfh_form']
		dust_ext_law = header_samplers['dust_ext_law']
		duste_switch = header_samplers['duste_stat']
		add_neb_emission = int(header_samplers['add_neb_emission'])
		add_agn = header_samplers['add_agn']
		add_igm_absorption = header_samplers['add_igm_absorption']
		if add_igm_absorption == 1:
			igm_type = int(header_samplers['igm_type'])
		elif add_igm_absorption == 0:
			igm_type = 0

		if duste_switch == 'duste':
			if 'dust_index' in header_samplers:
				def_params_val['dust_index'] = float(header_samplers['dust_index'])

		# redshift
		free_z = int(header_samplers['free_z'])
		if free_z == 0:
			gal_z = float(header_samplers['gal_z'])
			def_params_val['z'] = gal_z

		# cosmology parameter
		cosmo = header_samplers['cosmo']
		H0 = float(header_samplers['H0'])
		Om0 = float(header_samplers['Om0'])

		# get list parameters
		nparams0 = int(header_samplers['nparams'])
		params = []
		for pp in range(0,nparams0):
			str_temp = 'param%d' % pp
			params.append(header_samplers[str_temp])
		params.append('log_mass')
		nparams = nparams0 + 1

		# get best-fit parameters
		idx, min_val = min(enumerate(data_samplers['chi2']), key=operator.itemgetter(1))
		bfit_chi2 = data_samplers['chi2'][idx]
		bfit_params = {}
		for pp in range(0,nparams):
			bfit_params[params[pp]] = data_samplers[params[pp]][idx]

		# call fsps
		global sp
		sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)

		# generate the spectrum
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = bfit_params[params[pp]]

		spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
								add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
								igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,sfh_form=sfh_form,funit=funit)

		# get the photometric SED:
		bfit_photo_SED = filtering(spec_SED['wave'],spec_SED['flux_total'],filters)

		if wunit==0 or wunit=='angstrom':
			spec_wave = spec_SED['wave']
		elif wunit==1 or wunit=='micron':
			spec_wave = spec_SED['wave']/1.0e+4


	elif store_full_samplers == 0:
		#data_bfit_spec = hdu[2].data
		#data_bfit_photo = hdu[3].data

		# get best-fit spectrum
		if wunit==0 or wunit=='angstrom':
			spec_wave = data_bfit_spec['wave']
		elif wunit==1 or wunit=='micron':
			spec_wave = data_bfit_spec['wave']/1.0e+4

		spec_SED = {}
		spec_SED['wave'] = []
		spec_SED['flux_total'] = []
		spec_SED['flux_stellar'] = []
		spec_SED['flux_nebe'] = []
		spec_SED['flux_duste'] = []
		spec_SED['flux_agn'] = []

		spec_SED['flux_total'] = data_bfit_spec['flux_total']
		spec_SED['flux_stellar'] = data_bfit_spec['flux_stellar']
		if header_samplers['add_neb_emission'] == 1:
			spec_SED['flux_nebe'] = data_bfit_spec['flux_nebe']
		if header_samplers['duste_stat'] == 1:
			spec_SED['flux_duste'] = data_bfit_spec['flux_duste']
		if header_samplers['add_agn'] == 1:
			spec_SED['flux_agn'] = data_bfit_spec['flux_agn']

		# get best-fit photometric SED
		bfit_photo_SED = data_bfit_photo['flux']


	# plotting:
	fig1 = plt.figure(figsize=(14,7))
	f1 = plt.subplot()
	if logscale_y == True:
		f1.set_yscale('log')
	if logscale_x == True:
		f1.set_xscale('log')
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

	if wunit==0 or wunit=='angstrom':
		plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
	elif wunit==1 or wunit=='micron':
		plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

	if funit=='erg/s/cm2/A' or funit==0:
		plt.ylabel(r'$F_{\lambda}$ [erg $s^{-1}cm^{-2}\AA^{-1}$]', fontsize=int(fontsize_label))
	elif funit=='erg/s/cm2' or funit==1:
		plt.ylabel(r'$\lambda F_{\lambda}$ [erg $s^{-1}cm^{-2}$]', fontsize=int(fontsize_label))
	elif funit=='Jy' or funit==2:
		plt.ylabel(r'$F_{\nu}$ [Jy]', fontsize=int(fontsize_label))
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if xticks != None:
		plt.xticks(xticks)

	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())
	if xrange == None:
		if wunit==0 or wunit=='angstrom':
			plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
		elif wunit==1 or wunit=='micron':
			plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
	elif xrange != None:
		plt.xlim(xrange[0],xrange[1])

	# Convert unit of observed SED:
	if funit=='erg/s/cm2/A' or funit==0:
		obs_fluxes = obs_fluxes
		obs_flux_err = obs_flux_err
	elif funit=='erg/s/cm2' or funit==1:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)
	elif funit=='Jy' or funit==2:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if yrange == None:
		plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
	if yrange != None:
		plt.ylim(yrange[0],yrange[1])


	# alocate arrays:
	spec_total = spec_SED['flux_total']
	spec_stellar = []
	spec_nebe = []
	spec_duste = []
	spec_agn = []

	if decompose==1 or decompose==True:
		spec_stellar = spec_SED['flux_stellar']
		spec_nebe = spec_SED['flux_nebe']
		spec_duste = spec_SED['flux_duste']
		spec_agn = spec_SED['flux_agn']

		# stellar emission
		plt.plot(spec_wave,spec_stellar,lw=lw,color='darkorange',label='stellar emission')
		if add_neb_emission == 1:
			# nebular emission
			plt.plot(spec_wave,spec_nebe,lw=lw,color='darkcyan',label='nebular emission')
		if duste_switch == 1 or duste_switch == 'duste':
			# dust emission
			plt.plot(spec_wave,spec_duste,lw=lw,color='darkred',label='dust emission')
		if add_agn == 1:
			# AGN dusty torus emission
			plt.plot(spec_wave,spec_agn,lw=lw,color='darkgreen',label='AGN torus emission')

		# total
		plt.plot(spec_wave,spec_total,lw=lw,color='black',zorder=11,label='total')

		plt.legend(fontsize=int(fontsize_legend), loc=2, ncol=2)

	elif decompose==0 or decompose==False:
		# total
		plt.plot(spec_wave,spec_total,lw=lw,color='black',zorder=9)

	# plot best-fit model photometric SED and observed SED:
	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,bfit_photo_SED, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9)

		plt.errorbar(photo_cwave,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave,obs_fluxes, s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,bfit_photo_SED, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9)

		plt.errorbar(photo_cwave/1.0e+4,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave/1.0e+4,obs_fluxes, s=markersize, marker='s', lw=3, edgecolor=photo_color, color='none', zorder=11)

	f1.text(0.25, 0.9, "reduced $\chi^2 = %.3f$" % (bfit_chi2/nbands),verticalalignment='bottom', horizontalalignment='right',
	        transform=f1.transAxes,color='black', fontsize=20)	

	plt.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.98)

	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "sed_%s.png" % (name_sampler_fits1)
		
	plt.savefig(name_plot)

	return name_plot,spec_wave,spec_total,spec_stellar,spec_nebe,spec_duste,spec_agn



def old_plot_SED_rdsps(name_sampler_fits=None, logscale_x=True, logscale_y=True, xrange=None, yrange=None, wunit='micron', funit='erg/s/cm2/A', 
	decompose=1, plot_true=0, true_params = {'log_sfr': -99.0,'log_mass': -99.0,'log_dustmass':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,
	'log_qpah': -99.0,'log_umin': -99.0,'log_gamma': -99.0,'dust1':-99.0,'dust2': -99.0, 'dust_index':-99.0,'log_mw_age':-99.0,'log_age': -99.0, 
	'log_alpha':-99.0,'log_beta':-99.0, 'log_t0': -99.0, 'log_tau': -99.0,'logzsol': -99.0,'z': -99.0}, xticks=None, photo_color='red', 
	fontsize_tick=18, fontsize_label=25, fontsize_legend=18, markersize=100, lw=2.0, name_plot=None):

	"""Function for producing an SED plot from a fitting result obtained with the RDSPS method. 
	In this case, the best-fit model SED in the plot is the one with lowest chi-square from the input set of pre-calculated model SEDs in the fitting. 

	:param name_sampler_fits: (Mandatory, default: None)
		Name of input FITS file containing model SEDs and their probabilities. This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.

	:param logscale_x: (optional, default: True)
		Flag stating whether the x-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param logscale_y: (optional, default: True)
		Flag stating whether the y-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param xrange: (optional, default: None)
		Range in x-axis. The accepted format is: [xmin,xmax]. If xrange=None, the range will be defined based on 
		the wavelength range covered by the observed photometric SED.

	:param yrange: (optional, default: None)
		Range in y-axis. The accepted format is: [ymin,ymax]. If yrange=None, the range will be defined based on
		the fluxes range covered by the observed photometric SED.  
	
	:param wunit: (default: 'micron')
		Wavelength unit. Options are: (1)0 or 'angstrom' for Angstrom unit and (2)1 or 'micron' for micron unit.

	:param funit: (default: 'erg/s/cm2/A')
		Flux unit. Options are: (1)0 or 'erg/s/cm2/A', (2)1 or 'erg/s/cm2', and (3)2 or 'Jy'.

	:param decompose: (default: 1)
		Flag stating whether the best-fit model SED is broken-down into its components (value: 1 or True) or not (value: 0 or False).

	:param plot_true: (optional, default: 0)
		Flag stating whether to plot true model SED (in case available) or not. Options are: (1)0 or False and (2)1 or True.

	:param true_params: (optional)
		True values of parameters in case available. It should be in a dictionary format as shown in the default set. Only releavant if plot_true=1.

	:param xticks: (optional, default: None)
		List of ticks values in x-axis. If None, the default from matplotlib is used. If xticks is not None, the accepted input is in list format  
		xticks = []. The unit should be the same as the input wunit.

	:param photo_color: (optional, default: 'red')
		Color of photometric fluxes points (in string). The accepted input is the same as that available in matplotlib.

	:param fontsize_tick: (optional, default: 20)
		Fontsize for the tick. Only relevant if xticks is not None. 
	
	:param fontsize_label: (optional, default: 25)
		Fontsize for the x- and y-axis labels. 

	:param fontsize_legend: (optional, default: 18)
		Fontsize for the legend.

	:param markersize: (optional, default: 100)
		Size of the maarkers associated with the observed and model SEDs.

	:param lw: (optional, default: 1)
		Line width of the model SEDs.
	
	:param name_plot: (optional, default: None)
		Name of the output plot. This is optional parameter.

		
	
	:returns name_plot:
		Name of the output plot.

	:returns spec_wave:
		Wavelength grids of the total best-fit model spectrum.

	:returns spec_total:
		Fluxes grids of the total best-fit model spectrum.

	:returns spec_stellar:
		Stellar emission component of the best-fit model spectrum.

	:returns spec_nebe:
		Nebular emission component of the best-fit model spectrum.

	:returns spec_duste:
		Dust emission component of the best-fit model spectrum.

	:returns spec_agn:
		AGN dusty torus emission component of the best-fit model spectrum.
	"""

	def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta', 'log_age','dust_index','dust1','dust2',
					'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

	def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

	# open the FITS file:
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	data_samplers = hdu[1].data
	hdu.close()

	# some parameters 
	imf = int(header_samplers['imf'])
	sfh_form = header_samplers['sfh_form']
	dust_ext_law = header_samplers['dust_ext_law']
	duste_switch = header_samplers['duste_stat']
	add_neb_emission = int(header_samplers['add_neb_emission'])
	add_agn = header_samplers['add_agn']
	add_igm_absorption = header_samplers['add_igm_absorption']
	if add_igm_absorption == 1:
		igm_type = int(header_samplers['igm_type'])
	elif add_igm_absorption == 0:
		igm_type = 0

	if duste_switch == 'duste':
		if 'dust_index' in header_samplers:
			def_params_val['dust_index'] = float(header_samplers['dust_index'])

	# redshift
	free_z = int(header_samplers['free_z'])
	if free_z == 0:
		gal_z = float(header_samplers['gal_z'])
		def_params_val['z'] = gal_z

	# cosmology parameter
	cosmo = header_samplers['cosmo']
	H0 = float(header_samplers['H0'])
	Om0 = float(header_samplers['Om0'])

	# filters and observed SED
	nbands = int(header_samplers['nfilters'])
	filters = []
	obs_fluxes = np.zeros(nbands)
	obs_flux_err = np.zeros(nbands)
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		filters.append(header_samplers[str_temp])
		str_temp = 'flux%d' % bb
		obs_fluxes[bb] = float(header_samplers[str_temp])
		str_temp = 'flux_err%d' % bb
		obs_flux_err[bb] = float(header_samplers[str_temp])

	# central wavelength of all filters
	#photo_cwave = filtering.cwave_filters(filters)
	photo_cwave = cwave_filters(filters)

	# get list parameters
	nparams0 = int(header_samplers['nparams'])
	params = []
	for pp in range(0,nparams0):
		str_temp = 'param%d' % pp
		params.append(header_samplers[str_temp])
	params.append('log_mass')
	nparams = nparams0 + 1

	# get best-fit parameters
	idx, min_val = min(enumerate(data_samplers['chi2']), key=operator.itemgetter(1))
	bfit_chi2 = data_samplers['chi2'][idx]
	bfit_params = {}
	for pp in range(0,nparams):
		bfit_params[params[pp]] = data_samplers[params[pp]][idx]

	# call fsps
	global sp
	sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)

	# generate the spectrum
	params_val = def_params_val
	for pp in range(0,nparams):
		params_val[params[pp]] = bfit_params[params[pp]]

	spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
							add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
							igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,sfh_form=sfh_form,funit=funit)

	# get the photometric SED:
	bfit_photo_SED = filtering(spec_SED['wave'],spec_SED['flux_total'],filters)

	if wunit==0 or wunit=='angstrom':
		spec_wave = spec_SED['wave']
	elif wunit==1 or wunit=='micron':
		spec_wave = spec_SED['wave']/1.0e+4

	# plotting:
	fig1 = plt.figure(figsize=(14,7))
	f1 = plt.subplot()
	if logscale_y == True:
		f1.set_yscale('log')
	if logscale_x == True:
		f1.set_xscale('log')
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

	if wunit==0 or wunit=='angstrom':
		plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
	elif wunit==1 or wunit=='micron':
		plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

	if funit=='erg/s/cm2/A' or funit==0:
		plt.ylabel(r'$F_{\lambda}$ [erg $s^{-1}cm^{-2}\AA^{-1}$]', fontsize=int(fontsize_label))
	elif funit=='erg/s/cm2' or funit==1:
		plt.ylabel(r'$\lambda F_{\lambda}$ [erg $s^{-1}cm^{-2}$]', fontsize=int(fontsize_label))
	elif funit=='Jy' or funit==2:
		plt.ylabel(r'$F_{\nu}$ [Jy]', fontsize=int(fontsize_label))
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if xticks != None:
		plt.xticks(xticks)

	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())
	if xrange == None:
		if wunit==0 or wunit=='angstrom':
			plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
		elif wunit==1 or wunit=='micron':
			plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
	elif xrange != None:
		plt.xlim(xrange[0],xrange[1])

	# Convert unit of observed SED:
	if funit=='erg/s/cm2/A' or funit==0:
		obs_fluxes = obs_fluxes
		obs_flux_err = obs_flux_err
	elif funit=='erg/s/cm2' or funit==1:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)
	elif funit=='Jy' or funit==2:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if yrange == None:
		plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
	if yrange != None:
		plt.ylim(yrange[0],yrange[1])


	# alocate arrays:
	spec_total = spec_SED['flux_total']
	spec_stellar = []
	spec_nebe = []
	spec_duste = []
	spec_agn = []

	if decompose==1 or decompose==True:
		spec_stellar = spec_SED['flux_stellar']
		spec_nebe = spec_SED['flux_nebe']
		spec_duste = spec_SED['flux_duste']
		spec_agn = spec_SED['flux_agn']

		# stellar emission
		plt.plot(spec_wave,spec_stellar,lw=lw,color='darkorange',label='stellar emission')
		if add_neb_emission == 1:
			# nebular emission
			plt.plot(spec_wave,spec_nebe,lw=lw,color='darkcyan',label='nebular emission')
		if duste_switch == 1 or duste_switch == 'duste':
			# dust emission
			plt.plot(spec_wave,spec_duste,lw=lw,color='darkred',label='dust emission')
		if add_agn == 1:
			# AGN dusty torus emission
			plt.plot(spec_wave,spec_agn,lw=lw,color='darkgreen',label='AGN torus emission')

		# total
		plt.plot(spec_wave,spec_total,lw=lw,color='black',zorder=11,label='total')

		plt.legend(fontsize=int(fontsize_legend), loc=2, ncol=2)

	elif decompose==0 or decompose==False:
		# total
		plt.plot(spec_wave,spec_total,lw=lw,color='black',zorder=9)

	# plot true SED if required:
	if plot_true==1 or plot_true==True:
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = true_params[params[pp]]

		spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
							add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
							igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,sfh_form=sfh_form,funit=funit)

		if wunit==0 or wunit=='angstrom':
			wave0 = spec_SED['wave']
		elif wunit==1 or wunit=='micron':
			wave0 = spec_SED['wave']/1.0e+4

		if decompose==1 or decompose==True:
			# stellar emission
			plt.plot(wave0,spec_SED['flux_stellar'],lw=lw,color='darkorange',linestyle='--')
			# nebular emission
			plt.plot(wave0,spec_SED['flux_nebe'],lw=lw,color='darkcyan',linestyle='--')
			# dust emission
			plt.plot(wave0,spec_SED['flux_duste'],lw=lw,color='darkred',linestyle='--')
			# AGN dusty torus emission
			plt.plot(wave0,spec_SED['flux_agn'],lw=lw,color='darkgreen',linestyle='--')

		# total:
		plt.plot(wave0,spec_SED['flux_total'],lw=lw,color='black',linestyle='--',zorder=10)

	# plot best-fit model photometric SED and observed SED:
	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,bfit_photo_SED, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9)

		plt.errorbar(photo_cwave,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave,obs_fluxes, s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,bfit_photo_SED, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9)

		plt.errorbar(photo_cwave/1.0e+4,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave/1.0e+4,obs_fluxes, s=markersize, marker='s', lw=3, edgecolor=photo_color, color='none', zorder=11)

	f1.text(0.25, 0.9, "reduced $\chi^2 = %.3f$" % (bfit_chi2/nbands),verticalalignment='bottom', horizontalalignment='right',
	        transform=f1.transAxes,color='black', fontsize=20)	

	plt.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.98)

	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "sed_%s.png" % (name_sampler_fits1)
		
	plt.savefig(name_plot)

	return name_plot,spec_wave,spec_total,spec_stellar,spec_nebe,spec_duste,spec_agn




def plot_SED_rdsps_with_residual(name_sampler_fits=None, logscale_x=True, logscale_y=True, xrange=None, yrange=None, 
	wunit='micron', funit='erg/s/cm2/A', decompose=1, xticks=None, photo_color='red', residual_range=[-1.0,1.0], 
	fontsize_tick=18, fontsize_label=25, fontsize_legend=18, markersize=100, lw=2.0, name_plot=None):

	"""Function for producing an SED plot from a fitting result obtained with the RDSPS method. The output plot inludes residuals between the observed SED and best-fit model SED.  
	In this case, the best-fit model SED in the plot is the one with lowest chi-square from the input set of pre-calculated model SEDs in the fitting. 

	:param name_sampler_fits: (Mandatory, default: None)
		Name of input FITS file containing model SEDs and their probabilities. This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.

	:param logscale_x: (optional, default: True)
		Flag stating whether the x-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param logscale_y: (optional, default: True)
		Flag stating whether the y-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param xrange: (optional, default: None)
		Range in x-axis. The accepted format is: [xmin,xmax]. If xrange=None, the range will be defined based on 
		the wavelength range covered by the observed photometric SED.

	:param yrange: (optional, default: None)
		Range in y-axis. The accepted format is: [ymin,ymax]. If yrange=None, the range will be defined based on
		the fluxes range covered by the observed photometric SED.  
	
	:param wunit: (default: 'micron')
		Wavelength unit. Options are: (1)0 or 'angstrom' for Angstrom unit and (2)1 or 'micron' for micron unit.

	:param funit: (default: 'erg/s/cm2/A')
		Flux unit. Options are: (1)0 or 'erg/s/cm2/A', (2)1 or 'erg/s/cm2', and (3)2 or 'Jy'.

	:param decompose: (default: 1)
		Flag stating whether the best-fit model SED is broken-down into its components (value: 1 or True) or not (value: 0 or False).

	:param xticks: (optional, default: None)
		List of ticks values in x-axis. If None, the default from matplotlib is used. If xticks is not None, the accepted input is in list format  
		xticks = []. The unit should be the same as the input wunit.

	:param photo_color: (optional, default: 'red')
		Color of photometric fluxes points (in string). The accepted input is the same as that available in matplotlib.

	:param residual_range: (default: [-1.0,1.0])
		Residuals between observed SED and the median posterior model SED. 
		The residual in each band is defined as (f_D - f_M)/f_D, where f_D is flux in observed SED and f_M is flux in model SED.

	:param fontsize_tick: (optional, default: 20)
		Fontsize for the tick. Only relevant if xticks is not None. 
	
	:param fontsize_label: (optional, default: 25)
		Fontsize for the x- and y-axis labels. 

	:param fontsize_legend: (optional, default: 18)
		Fontsize for the legend.

	:param markersize: (optional, default: 100)
		Size of the maarkers associated with the observed and model SEDs.

	:param lw: (optional, default: 1)
		Line width of the model SEDs.
	
	:param name_plot: (optional, default: None)
		Name of the output plot. This is optional parameter.
	

	:returns name_plot:
		Name of the output plot.

	:returns spec_wave:
		Wavelength grids of the total best-fit model spectrum.

	:returns spec_total:
		Fluxes grids of the total best-fit model spectrum.

	:returns spec_stellar:
		Stellar emission component of the best-fit model spectrum.

	:returns spec_nebe:
		Nebular emission component of the best-fit model spectrum.

	:returns spec_duste:
		Dust emission component of the best-fit model spectrum.

	:returns spec_agn:
		AGN dusty torus emission component of the best-fit model spectrum.

	:returns residuals:
		Residuals.
	"""

	from matplotlib.gridspec import GridSpec

	def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta', 'log_age','dust_index','dust1','dust2',
					'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

	def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

	# open the FITS file:
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	if header_samplers['col1'] == 'rows':
		store_full_samplers = 0
		data_bfit_spec = hdu[2].data
		data_bfit_photo = hdu[3].data
	elif header_samplers['col1'] == 'id':
		store_full_samplers = 1
		data_samplers = hdu[1].data
	hdu.close()

	# filters and observed SED
	nbands = int(header_samplers['nfilters'])
	filters = []
	obs_fluxes = np.zeros(nbands)
	obs_flux_err = np.zeros(nbands)
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		filters.append(header_samplers[str_temp])
		str_temp = 'flux%d' % bb
		obs_fluxes[bb] = float(header_samplers[str_temp])
		str_temp = 'flux_err%d' % bb
		obs_flux_err[bb] = float(header_samplers[str_temp])

	# central wavelength of all filters
	photo_cwave = cwave_filters(filters)


	if store_full_samplers == 1:
		# some parameters 
		imf = int(header_samplers['imf'])
		sfh_form = header_samplers['sfh_form']
		dust_ext_law = header_samplers['dust_ext_law']
		duste_switch = header_samplers['duste_stat']
		add_neb_emission = int(header_samplers['add_neb_emission'])
		add_agn = header_samplers['add_agn']
		add_igm_absorption = header_samplers['add_igm_absorption']
		if add_igm_absorption == 1:
			igm_type = int(header_samplers['igm_type'])
		elif add_igm_absorption == 0:
			igm_type = 0

		if duste_switch == 'duste':
			if 'dust_index' in header_samplers:
				def_params_val['dust_index'] = float(header_samplers['dust_index'])

		# redshift
		free_z = int(header_samplers['free_z'])
		if free_z == 0:
			gal_z = float(header_samplers['gal_z'])
			def_params_val['z'] = gal_z

		# cosmology parameter
		cosmo = header_samplers['cosmo']
		H0 = float(header_samplers['H0'])
		Om0 = float(header_samplers['Om0'])

		# get list parameters
		nparams0 = int(header_samplers['nparams'])
		params = []
		for pp in range(0,nparams0):
			str_temp = 'param%d' % pp
			params.append(header_samplers[str_temp])
		params.append('log_mass')
		nparams = nparams0 + 1

		# get best-fit parameters
		idx, min_val = min(enumerate(data_samplers['chi2']), key=operator.itemgetter(1))
		bfit_chi2 = data_samplers['chi2'][idx]
		bfit_params = {}
		for pp in range(0,nparams):
			bfit_params[params[pp]] = data_samplers[params[pp]][idx]

		# call fsps
		global sp
		sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)

		# generate the spectrum
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = bfit_params[params[pp]]

		spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
								add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
								igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,sfh_form=sfh_form,funit=funit)

		# get the photometric SED:
		bfit_photo_SED = filtering(spec_SED['wave'],spec_SED['flux_total'],filters)

		if wunit==0 or wunit=='angstrom':
			spec_wave = spec_SED['wave']
		elif wunit==1 or wunit=='micron':
			spec_wave = spec_SED['wave']/1.0e+4

	elif store_full_samplers == 0:
		# get best-fit spectrum
		if wunit==0 or wunit=='angstrom':
			spec_wave = data_bfit_spec['wave']
		elif wunit==1 or wunit=='micron':
			spec_wave = data_bfit_spec['wave']/1.0e+4

		spec_SED = {}
		spec_SED['wave'] = []
		spec_SED['flux_total'] = []
		spec_SED['flux_stellar'] = []
		spec_SED['flux_nebe'] = []
		spec_SED['flux_duste'] = []
		spec_SED['flux_agn'] = []

		spec_SED['flux_total'] = data_bfit_spec['flux_total']
		spec_SED['flux_stellar'] = data_bfit_spec['flux_stellar']
		if header_samplers['add_neb_emission'] == 1:
			spec_SED['flux_nebe'] = data_bfit_spec['flux_nebe']
		if header_samplers['duste_stat'] == 1:
			spec_SED['flux_duste'] = data_bfit_spec['flux_duste']
		if header_samplers['add_agn'] == 1:
			spec_SED['flux_agn'] = data_bfit_spec['flux_agn']

		# get best-fit photometric SED
		bfit_photo_SED = data_bfit_photo['flux']


	# plotting
	fig1 = plt.figure(figsize=(14,7))

	gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], left=0.1, right=0.98, top=0.98, bottom=0.13, hspace=0.001)

	f1 = fig1.add_subplot(gs[0])
	plt.setp(f1.get_xticklabels(), visible=False)

	if logscale_y == True:
		f1.set_yscale('log')
	
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))

	if wunit==0 or wunit=='angstrom':
		plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
	elif wunit==1 or wunit=='micron':
		plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

	if funit=='erg/s/cm2/A' or funit==0:
		plt.ylabel(r'$F_{\lambda}$ [erg $s^{-1}cm^{-2}\AA^{-1}$]', fontsize=int(fontsize_label))
	elif funit=='erg/s/cm2' or funit==1:
		plt.ylabel(r'$\lambda F_{\lambda}$ [erg $s^{-1}cm^{-2}$]', fontsize=int(fontsize_label))
	elif funit=='Jy' or funit==2:
		plt.ylabel(r'$F_{\nu}$ [Jy]', fontsize=int(fontsize_label))
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if xticks != None:
		plt.xticks(xticks)

	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())
	if xrange == None:
		if wunit==0 or wunit=='angstrom':
			plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
		elif wunit==1 or wunit=='micron':
			plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
	elif xrange != None:
		plt.xlim(xrange[0],xrange[1])

	# Convert unit of observed SED:
	if funit=='erg/s/cm2/A' or funit==0:
		obs_fluxes = obs_fluxes
		obs_flux_err = obs_flux_err
	elif funit=='erg/s/cm2' or funit==1:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)
	elif funit=='Jy' or funit==2:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if yrange == None:
		plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
	if yrange != None:
		plt.ylim(yrange[0],yrange[1])


	# alocate arrays:
	spec_total = spec_SED['flux_total']
	spec_stellar = []
	spec_nebe = []
	spec_duste = []
	spec_agn = []

	if decompose==1 or decompose==True:
		spec_stellar = spec_SED['flux_stellar']
		spec_nebe = spec_SED['flux_nebe']
		spec_duste = spec_SED['flux_duste']
		spec_agn = spec_SED['flux_agn']

		# stellar emission
		plt.plot(spec_wave,spec_stellar,lw=lw,color='darkorange',label='stellar emission')
		if add_neb_emission == 1:
			# nebular emission
			plt.plot(spec_wave,spec_nebe,lw=lw,color='darkcyan',label='nebular emission')
		if duste_switch == 1 or duste_switch == 'duste':
			# dust emission
			plt.plot(spec_wave,spec_duste,lw=lw,color='darkred',label='dust emission')
		if add_agn == 1:
			# AGN dusty torus emission
			plt.plot(spec_wave,spec_agn,lw=lw,color='darkgreen',label='AGN torus emission')

		# total
		plt.plot(spec_wave,spec_total,lw=lw,color='black',zorder=11,label='total')

		plt.legend(fontsize=int(fontsize_legend), loc=2, ncol=2)

	elif decompose==0 or decompose==False:
		# total
		plt.plot(spec_wave,spec_total,lw=lw,color='black',zorder=9)


	# plot observed SED
	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,bfit_photo_SED, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9)

		plt.errorbar(photo_cwave,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave,obs_fluxes, s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,bfit_photo_SED, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9)

		plt.errorbar(photo_cwave/1.0e+4,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave/1.0e+4,obs_fluxes, s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)

	f1.text(0.25, 0.9, "reduced $\chi^2 = %.3f$" % (bfit_chi2/nbands),verticalalignment='bottom', horizontalalignment='right',
	       transform=f1.transAxes,color='black', fontsize=20)

	# plot residual
	f1 = fig1.add_subplot(gs[1])
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

	if logscale_x == True:
		f1.set_xscale('log')

	plt.ylabel(r'residual', fontsize=25)
	plt.ylim(residual_range[0],residual_range[1])
	if wunit==0 or wunit=='angstrom':
		plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
	elif wunit==1 or wunit=='micron':
		plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

	if xticks != None:
		plt.xticks(xticks)
	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	if xrange == None:
		if wunit==0 or wunit=='angstrom':
			plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
			xmin = min(photo_cwave)*0.7
			xmax = max(photo_cwave)*1.3
		elif wunit==1 or wunit=='micron':
			plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
			xmin = min(photo_cwave)*0.7/1e+4
			xmax = max(photo_cwave)*1.3/1e+4
	elif xrange != None:
		plt.xlim(xrange[0],xrange[1])
		xmin = xrange[0]
		xmax = xrange[1]

	# get residual:
	residuals = (obs_fluxes-bfit_photo_SED)/obs_fluxes

	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,residuals, s=80, marker='s', lw=3.0, 
							color='gray', zorder=9, alpha=1.0)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,residuals, s=80, marker='s', lw=3.0, 
							color='gray', zorder=9, alpha=1.0)

	x = np.linspace(xmin,xmax,100)
	y = x-x
	plt.plot(x,y,lw=2,color='black',linestyle='--')	

	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "sed_%s.png" % (name_sampler_fits1)
		
	plt.savefig(name_plot)

	return name_plot,spec_wave,spec_total,spec_stellar,spec_nebe,spec_duste,spec_agn,residuals




def old_plot_SED_rdsps_with_residual(name_sampler_fits=None, logscale_x=True, logscale_y=True, xrange=None, yrange=None, wunit='micron', funit='erg/s/cm2/A', 
	decompose=1, plot_true=0, true_params = {'log_sfr': -99.0,'log_mass': -99.0,'log_dustmass':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,
	'log_qpah': -99.0,'log_umin': -99.0,'log_gamma': -99.0,'dust1':-99.0,'dust2': -99.0, 'dust_index':-99.0,'log_mw_age':-99.0,'log_age': -99.0, 
	'log_alpha':-99.0,'log_beta':-99.0, 'log_t0': -99.0, 'log_tau': -99.0,'logzsol': -99.0,'z': -99.0}, xticks=None, photo_color='red', residual_range=[-1.0,1.0],
	fontsize_tick=18, fontsize_label=25, fontsize_legend=18, markersize=100, lw=2.0, name_plot=None):

	"""Function for producing an SED plot from a fitting result obtained with the RDSPS method. The output plot inludes residuals between the observed SED and best-fit model SED.  
	In this case, the best-fit model SED in the plot is the one with lowest chi-square from the input set of pre-calculated model SEDs in the fitting. 

	:param name_sampler_fits: (Mandatory, default: None)
		Name of input FITS file containing model SEDs and their probabilities. This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.

	:param logscale_x: (optional, default: True)
		Flag stating whether the x-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param logscale_y: (optional, default: True)
		Flag stating whether the y-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param xrange: (optional, default: None)
		Range in x-axis. The accepted format is: [xmin,xmax]. If xrange=None, the range will be defined based on 
		the wavelength range covered by the observed photometric SED.

	:param yrange: (optional, default: None)
		Range in y-axis. The accepted format is: [ymin,ymax]. If yrange=None, the range will be defined based on
		the fluxes range covered by the observed photometric SED.  
	
	:param wunit: (default: 'micron')
		Wavelength unit. Options are: (1)0 or 'angstrom' for Angstrom unit and (2)1 or 'micron' for micron unit.

	:param funit: (default: 'erg/s/cm2/A')
		Flux unit. Options are: (1)0 or 'erg/s/cm2/A', (2)1 or 'erg/s/cm2', and (3)2 or 'Jy'.

	:param decompose: (default: 1)
		Flag stating whether the best-fit model SED is broken-down into its components (value: 1 or True) or not (value: 0 or False).

	:param plot_true: (optional, default: 0)
		Flag stating whether to plot true model SED (in case available) or not. Options are: (1)0 or False and (2)1 or True.

	:param true_params: (optional)
		True values of parameters in case available. It should be in a dictionary format as shown in the default set. Only releavant if plot_true=1.

	:param xticks: (optional, default: None)
		List of ticks values in x-axis. If None, the default from matplotlib is used. If xticks is not None, the accepted input is in list format  
		xticks = []. The unit should be the same as the input wunit.

	:param photo_color: (optional, default: 'red')
		Color of photometric fluxes points (in string). The accepted input is the same as that available in matplotlib.

	:param residual_range: (default: [-1.0,1.0])
		Residuals between observed SED and the median posterior model SED. 
		The residual in each band is defined as (f_D - f_M)/f_D, where f_D is flux in observed SED and f_M is flux in model SED.

	:param fontsize_tick: (optional, default: 20)
		Fontsize for the tick. Only relevant if xticks is not None. 
	
	:param fontsize_label: (optional, default: 25)
		Fontsize for the x- and y-axis labels. 

	:param fontsize_legend: (optional, default: 18)
		Fontsize for the legend.

	:param markersize: (optional, default: 100)
		Size of the maarkers associated with the observed and model SEDs.

	:param lw: (optional, default: 1)
		Line width of the model SEDs.
	
	:param name_plot: (optional, default: None)
		Name of the output plot. This is optional parameter.
	

	:returns name_plot:
		Name of the output plot.

	:returns spec_wave:
		Wavelength grids of the total best-fit model spectrum.

	:returns spec_total:
		Fluxes grids of the total best-fit model spectrum.

	:returns spec_stellar:
		Stellar emission component of the best-fit model spectrum.

	:returns spec_nebe:
		Nebular emission component of the best-fit model spectrum.

	:returns spec_duste:
		Dust emission component of the best-fit model spectrum.

	:returns spec_agn:
		AGN dusty torus emission component of the best-fit model spectrum.

	:returns residuals:
		Residuals.
	"""

	from matplotlib.gridspec import GridSpec

	def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta', 'log_age','dust_index','dust1','dust2',
					'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

	#def_params_val={'log_mass':0.0,'z':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,'log_qpah':-99.0,'log_umin':-99.0,'log_gamma':-99.0,
	#				'dust1':-99.0,'dust2':-99.0, 'dust_index':-99.0,'log_age':-99.0,'log_alpha':-99.0,'log_beta':-99.0,
	#				'log_t0':-99.0,'log_tau':-99.0,'logzsol':-99.0}

	def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

	# open the FITS file:
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	data_samplers = hdu[1].data
	hdu.close()

	# some parameters 
	imf = int(header_samplers['imf'])
	sfh_form = header_samplers['sfh_form']
	dust_ext_law = header_samplers['dust_ext_law']
	duste_switch = header_samplers['duste_stat']
	add_neb_emission = int(header_samplers['add_neb_emission'])
	add_agn = header_samplers['add_agn']
	add_igm_absorption = header_samplers['add_igm_absorption']
	if add_igm_absorption == 1:
		igm_type = int(header_samplers['igm_type'])
	elif add_igm_absorption == 0:
		igm_type = 0

	if duste_switch == 'duste':
		if 'dust_index' in header_samplers:
			def_params_val['dust_index'] = float(header_samplers['dust_index'])

	# redshift
	free_z = int(header_samplers['free_z'])
	if free_z == 0:
		gal_z = float(header_samplers['gal_z'])
		def_params_val['z'] = gal_z

	# cosmology parameter
	cosmo = header_samplers['cosmo']
	H0 = float(header_samplers['H0'])
	Om0 = float(header_samplers['Om0'])

	# filters and observed SED
	nbands = int(header_samplers['nfilters'])
	filters = []
	obs_fluxes = np.zeros(nbands)
	obs_flux_err = np.zeros(nbands)
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		filters.append(header_samplers[str_temp])
		str_temp = 'flux%d' % bb
		obs_fluxes[bb] = float(header_samplers[str_temp])
		str_temp = 'flux_err%d' % bb
		obs_flux_err[bb] = float(header_samplers[str_temp])

	# central wavelength of all filters
	photo_cwave = cwave_filters(filters)

	# get list parameters
	nparams0 = int(header_samplers['nparams'])
	params = []
	for pp in range(0,nparams0):
		str_temp = 'param%d' % pp
		params.append(header_samplers[str_temp])
	params.append('log_mass')
	nparams = nparams0 + 1

	# get best-fit parameters
	idx, min_val = min(enumerate(data_samplers['chi2']), key=operator.itemgetter(1))
	bfit_chi2 = data_samplers['chi2'][idx]
	bfit_params = {}
	for pp in range(0,nparams):
		bfit_params[params[pp]] = data_samplers[params[pp]][idx]

	# call fsps
	global sp
	sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)

	# generate the spectrum
	params_val = def_params_val
	for pp in range(0,nparams):
		params_val[params[pp]] = bfit_params[params[pp]]

	spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
							add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
							igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,sfh_form=sfh_form,funit=funit)

	# get the photometric SED:
	bfit_photo_SED = filtering(spec_SED['wave'],spec_SED['flux_total'],filters)

	if wunit==0 or wunit=='angstrom':
		spec_wave = spec_SED['wave']
	elif wunit==1 or wunit=='micron':
		spec_wave = spec_SED['wave']/1.0e+4


	# plotting
	fig1 = plt.figure(figsize=(14,7))

	gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], left=0.1, right=0.98, top=0.98, 
					bottom=0.13, hspace=0.001)

	f1 = fig1.add_subplot(gs[0])
	plt.setp(f1.get_xticklabels(), visible=False)

	if logscale_y == True:
		f1.set_yscale('log')
	#if logscale_x == True:
	#	f1.set_xscale('log')
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	#plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

	if wunit==0 or wunit=='angstrom':
		plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
	elif wunit==1 or wunit=='micron':
		plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

	if funit=='erg/s/cm2/A' or funit==0:
		plt.ylabel(r'$F_{\lambda}$ [erg $s^{-1}cm^{-2}\AA^{-1}$]', fontsize=int(fontsize_label))
	elif funit=='erg/s/cm2' or funit==1:
		plt.ylabel(r'$\lambda F_{\lambda}$ [erg $s^{-1}cm^{-2}$]', fontsize=int(fontsize_label))
	elif funit=='Jy' or funit==2:
		plt.ylabel(r'$F_{\nu}$ [Jy]', fontsize=int(fontsize_label))
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if xticks != None:
		plt.xticks(xticks)

	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())
	if xrange == None:
		if wunit==0 or wunit=='angstrom':
			plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
		elif wunit==1 or wunit=='micron':
			plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
	elif xrange != None:
		plt.xlim(xrange[0],xrange[1])

	# Convert unit of observed SED:
	if funit=='erg/s/cm2/A' or funit==0:
		obs_fluxes = obs_fluxes
		obs_flux_err = obs_flux_err
	elif funit=='erg/s/cm2' or funit==1:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)
	elif funit=='Jy' or funit==2:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if yrange == None:
		plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
	if yrange != None:
		plt.ylim(yrange[0],yrange[1])


	# alocate arrays:
	spec_total = spec_SED['flux_total']
	spec_stellar = []
	spec_nebe = []
	spec_duste = []
	spec_agn = []

	if decompose==1 or decompose==True:
		spec_stellar = spec_SED['flux_stellar']
		spec_nebe = spec_SED['flux_nebe']
		spec_duste = spec_SED['flux_duste']
		spec_agn = spec_SED['flux_agn']

		# stellar emission
		plt.plot(spec_wave,spec_stellar,lw=lw,color='darkorange',label='stellar emission')
		if add_neb_emission == 1:
			# nebular emission
			plt.plot(spec_wave,spec_nebe,lw=lw,color='darkcyan',label='nebular emission')
		if duste_switch == 1 or duste_switch == 'duste':
			# dust emission
			plt.plot(spec_wave,spec_duste,lw=lw,color='darkred',label='dust emission')
		if add_agn == 1:
			# AGN dusty torus emission
			plt.plot(spec_wave,spec_agn,lw=lw,color='darkgreen',label='AGN torus emission')

		# total
		plt.plot(spec_wave,spec_total,lw=lw,color='black',zorder=11,label='total')

		plt.legend(fontsize=int(fontsize_legend), loc=2, ncol=2)

	elif decompose==0 or decompose==False:
		# total
		plt.plot(spec_wave,spec_total,lw=lw,color='black',zorder=9)

	# plot true SED if required:
	if plot_true==1 or plot_true==True:
		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = true_params[params[pp]]

		spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
							add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
							igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,sfh_form=sfh_form,funit=funit)

		if wunit==0 or wunit=='angstrom':
			wave0 = spec_SED['wave']
		elif wunit==1 or wunit=='micron':
			wave0 = spec_SED['wave']/1.0e+4

		if decompose==1 or decompose==True:
			# stellar emission
			plt.plot(wave0,spec_SED['flux_stellar'],lw=lw,color='darkorange',linestyle='--')
			# nebular emission
			plt.plot(wave0,spec_SED['flux_nebe'],lw=lw,color='darkcyan',linestyle='--')
			# dust emission
			plt.plot(wave0,spec_SED['flux_duste'],lw=lw,color='darkred',linestyle='--')
			# AGN dusty torus emission
			plt.plot(wave0,spec_SED['flux_agn'],lw=lw,color='darkgreen',linestyle='--')

		# total:
		plt.plot(wave0,spec_SED['flux_total'],lw=lw,color='black',linestyle='--',zorder=10)


	# plot observed SED
	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,bfit_photo_SED, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9)

		plt.errorbar(photo_cwave,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave,obs_fluxes, s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,bfit_photo_SED, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9)

		plt.errorbar(photo_cwave/1.0e+4,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave/1.0e+4,obs_fluxes, s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)

	f1.text(0.25, 0.9, "reduced $\chi^2 = %.3f$" % (bfit_chi2/nbands),verticalalignment='bottom', horizontalalignment='right',
	       transform=f1.transAxes,color='black', fontsize=20)

	# plot residual
	f1 = fig1.add_subplot(gs[1])
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

	if logscale_x == True:
		f1.set_xscale('log')

	plt.ylabel(r'residual', fontsize=25)
	plt.ylim(residual_range[0],residual_range[1])
	if wunit==0 or wunit=='angstrom':
		plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
	elif wunit==1 or wunit=='micron':
		plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

	if xticks != None:
		plt.xticks(xticks)
	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	if xrange == None:
		if wunit==0 or wunit=='angstrom':
			plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
			xmin = min(photo_cwave)*0.7
			xmax = max(photo_cwave)*1.3
		elif wunit==1 or wunit=='micron':
			plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
			xmin = min(photo_cwave)*0.7/1e+4
			xmax = max(photo_cwave)*1.3/1e+4
	elif xrange != None:
		plt.xlim(xrange[0],xrange[1])
		xmin = xrange[0]
		xmax = xrange[1]

	# get residual:
	residuals = (obs_fluxes-bfit_photo_SED)/obs_fluxes

	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,residuals, s=80, marker='s', lw=3.0, 
							color='gray', zorder=9, alpha=1.0)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,residuals, s=80, marker='s', lw=3.0, 
							color='gray', zorder=9, alpha=1.0)

	x = np.linspace(xmin,xmax,100)
	y = x-x
	plt.plot(x,y,lw=2,color='black',linestyle='--')	

	#plt.subplots_adjust(left=0.25, right=0.98, bottom=0.25, top=0.98)

	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "sed_%s.png" % (name_sampler_fits1)
		
	plt.savefig(name_plot)

	return name_plot,spec_wave,spec_total,spec_stellar,spec_nebe,spec_duste,spec_agn,residuals


def plot_SED_mcmc(name_sampler_fits=None, nchains=100, logscale_x=True, logscale_y=True, xrange=None, yrange=None, wunit='micron', 
	funit='erg/s/cm2/A', decompose=1, shadow_plot=1, add_neb_emission=None, cosmo=0, H0=70.0, Om0=0.3, gas_logu=-2.0, 
	xticks=None, photo_color='red', fontsize_tick=20, fontsize_label=25, fontsize_legend=18, markersize=100, 
	lw=1.0, name_plot=None):

	"""Function for producing an SED plot from a fitting result obtained with the MCMC method. 

	:param name_sampler_fits: (Mandatory, default: None)
		Name of input FITS file containing sampler chains from the MCMC fitting. This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions. 

	:param nchains: (default: 100)
		Number of randomly selected sampler chains to be used for calculating median posterior model SED.

	:param logscale_x: (optional, default: True)
		Flag stating whether the x-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param logscale_y: (optional, default: True)
		Flag stating whether the y-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param xrange: (optional, default: None)
		Range in x-axis. The accepted format is: [xmin,xmax]. If xrange=None, the range will be defined based on 
		the wavelength range covered by the observed photometric SED.

	:param yrange: (optional, default: None)
		Range in y-axis. The accepted format is: [ymin,ymax]. If yrange=None, the range will be defined based on
		the fluxes range covered by the observed photometric SED.  
	
	:param wunit: (default: 'micron')
		Wavelength unit. Options are: (1)0 or 'angstrom' for Angstrom unit and (2)1 or 'micron' for micron unit.

	:param funit: (default: 'erg/s/cm2/A')
		Flux unit. Options are: (1)0 or 'erg/s/cm2/A', (2)1 or 'erg/s/cm2', and (3)2 or 'Jy'.

	:param decompose: (default: 1)
		Flag stating whether the best-fit model SED is broken-down into its components (value: 1 or True) or not (value: 0 or False).

	:param shadow_plot: (default: 1)
		Switch for shadow plot (a plot in which uncertainty is plotted as transprent region around the median value). 
		Options are: (1)1 or True and (2)0 or False. If shadow_plot=0 or False, actual model SEDs are plotted.

	:param add_neb_emission: (optional, default: None)
		Flag stating whether to include emission lines. Options are: None, 0, and 1. If None, the decision whether to plot emission lines 
		or not is based on the `add_neb_emission` flag in the header of the `name_sampler_fits` (which tells whether the emission lines modeling
		is included in the fitting).

	:param cosmo: (default: 0)
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param xticks: (optional, default: None)
		List of ticks values in x-axis. If None, the default from matplotlib is used. If xticks is not None, the accepted input is in list format  
		xticks = []. The unit should be the same as the input wunit.

	:param photo_color: (optional, default: 'red')
		Color of photometric fluxes points (in string). The accepted input is the same as that available in matplotlib.

	:param fontsize_tick: (optional, default: 20)
		Fontsize for the tick. Only relevant if xticks is not None. 
	
	:param fontsize_label: (optional, default: 25)
		Fontsize for the x- and y-axis labels. 

	:param fontsize_legend: (optional, default: 18)
		Fontsize for the legend.

	:param markersize: (optional, default: 100)
		Size of the maarkers associated with the observed and model SEDs.

	:param lw: (optional, default: 1)
		Line width of the model SEDs.
	
	:param name_plot: (optional, default: None)
		Name of the output plot. This is optional parameter.


	:returns name_plot:
		Name of the output pot.

	:returns wave:
		Wavelength grids of the median posterior model SED.

	:returns p16_spec_tot:
		The 16th percentile posterior model spectroscopic SED.

	:returns p50_spec_tot:
		The 50th percentile posterior model spectroscopic SED.

	:returns p84_spec_tot:
		The 84th percentile posterior model spectroscopic SED.

	:returns photo_cwave:
		Central wavelengths of the photometric filters.

	:returns p50_photo_flux:
		Median posterior model photometric SED. 
	"""

	def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta', 'log_age','dust_index','dust1','dust2',
					'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

	def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

	# open the FITS file
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	data_samplers = hdu[1].data
	hdu.close()

	sfh_form = header_samplers['sfh_form']
	dust_ext_law = header_samplers['dust_ext_law']
	duste_switch = header_samplers['duste_stat'] 
	if add_neb_emission == None:
		add_neb_emission = int(header_samplers['add_neb_emission'])
	else:
		add_neb_emission = add_neb_emission

	if add_neb_emission == 1:
		if gas_logu == -2.0:
			if 'gas_logu' in header_samplers: 
				gas_logu = float(header_samplers['gas_logu'])
			else:
				gas_logu = -2.0

	# parameters in the fitting
	nparams = int(header_samplers['nparams'])
	params = []
	for pp in range(0,nparams):
		str_temp = 'param%d' % pp
		params.append(header_samplers[str_temp])

	# filters and observed fluxes
	nbands = int(header_samplers['nfilters'])
	filters = []
	obs_fluxes = np.zeros(nbands)
	obs_flux_err = np.zeros(nbands)
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		filters.append(header_samplers[str_temp])
		str_temp = "flux%d" % bb
		obs_fluxes[bb] = float(header_samplers[str_temp])
		str_temp = "flux_err%d" % bb
		obs_flux_err[bb] = float(header_samplers[str_temp])

	# central wavelength of all filters:
	#photo_cwave = filtering.cwave_filters(filters)
	photo_cwave = cwave_filters(filters)
	
	free_z = int(header_samplers['free_z'])
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

	if free_z == 0:
		gal_z = float(header_samplers['gal_z'])
		def_params_val['z'] = gal_z

	if duste_switch == 'duste':
		if 'dust_index' in header_samplers:
			def_params_val['dust_index'] = float(header_samplers['dust_index'])

	# cosmology parameter:
	if 'cosmo' in header_samplers:
		cosmo = header_samplers['cosmo']
		H0 = float(header_samplers['H0'])
		Om0 = float(header_samplers['Om0'])
	else:
		cosmo = cosmo
		H0 = H0
		Om0 = Om0

	# call fsps
	global sp
	sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)

	## exclude saturated samplers: log(SFR)~-29.99..
	idx_sel = np.where(data_samplers['log_sfr']>-29.0)

	#rand_idx = np.random.uniform(0,nsamplers,nchains)
	rand_idx = np.random.uniform(0,len(idx_sel[0]),nchains)

	rand_wave = []
	rand_spec_tot = []
	rand_spec_stellar = []
	rand_spec_duste = []
	rand_spec_agn = []
	rand_spec_nebe = []
	rand_photo_flux = []
	for ii in range(0,nchains):
		#idx = rand_idx[ii]
		idx = idx_sel[0][int(rand_idx[ii])]

		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = data_samplers[params[pp]][int(idx)]

		spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
								add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
								igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,sfh_form=sfh_form,funit=funit)

		if ii == 0:
			if wunit==0 or wunit=='angstrom':
				rand_wave = spec_SED['wave']
			elif wunit==1 or wunit=='micron':
				rand_wave = spec_SED['wave']/1.0e+4

		rand_spec_tot.append(spec_SED['flux_total'])
		rand_spec_stellar.append(spec_SED['flux_stellar'])

		if add_neb_emission == 1:
			rand_spec_nebe.append(spec_SED['flux_nebe'])
		if duste_switch == 1 or duste_switch=='duste':
			rand_spec_duste.append(spec_SED['flux_duste'])
		if add_agn == 1:
			rand_spec_agn.append(spec_SED['flux_agn'])

		# photometric SED:
		#mod_fluxes = filtering.filtering(spec_SED['wave'],spec_SED['flux_total'],filters)
		mod_fluxes = filtering(spec_SED['wave'],spec_SED['flux_total'],filters)
		rand_photo_flux.append(mod_fluxes)

		# end of for ii: nchains

	# plotting:
	fig1 = plt.figure(figsize=(14,7))
	f1 = plt.subplot()
	if logscale_y == True:
		f1.set_yscale('log')
	if logscale_x == True:
		f1.set_xscale('log')
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

	if wunit==0 or wunit=='angstrom':
		plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
	elif wunit==1 or wunit=='micron':
		plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

	if funit=='erg/s/cm2/A' or funit==0:
		plt.ylabel(r'$F_{\lambda}$ [erg $s^{-1}cm^{-2}\AA^{-1}$]', fontsize=int(fontsize_label))
	elif funit=='erg/s/cm2' or funit==1:
		plt.ylabel(r'$\lambda F_{\lambda}$ [erg $s^{-1}cm^{-2}$]', fontsize=int(fontsize_label))
	elif funit=='Jy' or funit==2:
		plt.ylabel(r'$F_{\nu}$ [Jy]', fontsize=int(fontsize_label))
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if xticks != None:
		plt.xticks(xticks)

	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())
	if xrange == None:
		if wunit==0 or wunit=='angstrom':
			plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
		elif wunit==1 or wunit=='micron':
			plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
	elif xrange != None:
		plt.xlim(xrange[0],xrange[1])

	# convert unit of observed SED:
	if funit=='erg/s/cm2/A' or funit==0:
		obs_fluxes = obs_fluxes
		obs_flux_err = obs_flux_err
	elif funit=='erg/s/cm2' or funit==1:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)
	elif funit=='Jy' or funit==2:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if yrange == None:
		plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
	if yrange != None:
		plt.ylim(yrange[0],yrange[1])

	p16_spec_tot = []
	p50_spec_tot = []
	p84_spec_tot = []
	if shadow_plot==1 or shadow_plot==True:
		# total:
		p16_spec_tot = np.percentile(rand_spec_tot,16,axis=0)
		p50_spec_tot = np.percentile(rand_spec_tot,50,axis=0)
		p84_spec_tot = np.percentile(rand_spec_tot,84,axis=0)
		f1.fill_between(rand_wave,p16_spec_tot,p84_spec_tot,facecolor='gray',alpha=0.5,zorder=9)
		if decompose==1 or decompose==True:
			plt.plot(rand_wave,p50_spec_tot,lw=lw,color='black',zorder=9,label='total')
		elif decompose==0 or decompose==False:
			plt.plot(rand_wave,p50_spec_tot,lw=lw,color='black',zorder=9)

		if decompose==1 or decompose==True:
			# stellar emission
			f1.fill_between(rand_wave,np.percentile(rand_spec_stellar,16,axis=0),np.percentile(rand_spec_stellar,84,axis=0),facecolor='orange',
									alpha=0.25,zorder=8)
			plt.plot(rand_wave,np.percentile(rand_spec_stellar,50,axis=0),lw=lw,color='darkorange',zorder=8,label='stellar emission')
			# nebular emission
			if add_neb_emission == 1:
				f1.fill_between(rand_wave,np.percentile(rand_spec_nebe,16,axis=0),np.percentile(rand_spec_nebe,84,axis=0),
									facecolor='cyan',alpha=0.25,zorder=8)
				plt.plot(rand_wave,np.percentile(rand_spec_nebe,50,axis=0),lw=lw,color='darkcyan',zorder=8,label='nebular emission')
			# dust emission
			if duste_switch == 1 or duste_switch == 'duste':
				f1.fill_between(rand_wave,np.percentile(rand_spec_duste,16,axis=0),np.percentile(rand_spec_duste,84,axis=0),facecolor='red',
									alpha=0.25,zorder=8)
				plt.plot(rand_wave,np.percentile(rand_spec_duste,50,axis=0),lw=lw,color='darkred',zorder=8,label='dust emission')
			# AGN dusty torus emission
			if add_agn == 1:
				f1.fill_between(rand_wave,np.percentile(rand_spec_agn,16,axis=0),np.percentile(rand_spec_agn,84,axis=0),facecolor='green',
									alpha=0.25,zorder=8)
				plt.plot(rand_wave,np.percentile(rand_spec_agn,50,axis=0),lw=lw,color='darkgreen',zorder=8,label='AGN torus emission')

			plt.legend(fontsize=int(fontsize_legend), loc=2, ncol=2)

	elif shadow_plot==0 or shadow_plot==False:
		if decompose==1 or decompose==True:
			for ii in range(0,nchains):
				# total
				plt.plot(rand_wave,rand_spec_tot[ii],lw=0.8,color='gray',alpha=0.6)
				# stellar
				plt.plot(rand_wave,rand_spec_stellar[ii],lw=0.5,color='orange',alpha=0.4)
				if add_neb_emission == 1:
					# nebular emission
					plt.plot(rand_wave,rand_spec_nebe[ii],lw=0.5,color='cyan',alpha=0.4)
				if duste_switch == 1 or duste_switch == 'duste':
					# dust emission
					plt.plot(rand_wave,rand_spec_duste[ii],lw=0.5,color='red',alpha=0.4)
				if add_agn == 1:
					# AGN
					plt.plot(rand_wave,rand_spec_agn[ii],lw=0.5,color='green',alpha=0.4)

		elif decompose==0 or decompose==False:
			for ii in range(0,nchains):
				# total
				plt.plot(rand_wave,rand_spec_tot[ii],lw=0.8,color='gray',alpha=0.6)

	p50_photo_flux = np.percentile(rand_photo_flux,50,axis=0)

	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,p50_photo_flux, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9,alpha=0.5)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,p50_photo_flux, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9,alpha=0.5)

	# plot observed SED:
	if wunit==0 or wunit=='angstrom':
		plt.errorbar(photo_cwave,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave,obs_fluxes, s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)
	elif wunit==1 or wunit=='micron':
		plt.errorbar(photo_cwave/1.0e+4,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave/1.0e+4,obs_fluxes, s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)

	plt.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.98)

	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "sed_%s.png" % (name_sampler_fits1)
		
	plt.savefig(name_plot)

	wave = rand_wave
	
	return name_plot,wave,p16_spec_tot,p50_spec_tot,p84_spec_tot,photo_cwave,p50_photo_flux


def plot_SED_mcmc_with_residual(name_sampler_fits=None, nchains=100, logscale_x=True, logscale_y=True, xrange=None, 
	yrange=None, wunit='micron', funit='erg/s/cm2/A', decompose=1, shadow_plot=1, cosmo=0, H0=70.0, Om0=0.3, gas_logu=-2.0, 
	add_neb_emission=None, xticks=None, photo_color='red', residual_range=[-1.0,1.0], fontsize_tick=18, 
	fontsize_label=28, fontsize_legend=20, markersize=100, lw=1.0, name_plot=None):
	
	"""Function for producing an SED plot from a fitting result obtained with the MCMC method.
	This function add residuals (between observed SED and models) in the bottom panel of the SED plot. 

	:param name_sampler_fits: (Mandatory, default: None)
		Name of input FITS file containing sampler chains from the MCMC fitting. This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.

	:param nchains: (default: 100)
		Number of randomly selected sampler chains to be used for calculating median posterior model SED.

	:param logscale_x: (optional, default: True)
		Flag stating whether the x-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param logscale_y: (optional, default: True)
		Flag stating whether the y-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param xrange: (optional, default: None)
		Range in x-axis. The accepted format is: [xmin,xmax]. If xrange=None, the range will be defined based on 
		the wavelength range covered by the observed photometric SED.

	:param yrange: (optional, default: None)
		Range in y-axis. The accepted format is: [ymin,ymax]. If yrange=None, the range will be defined based on
		the fluxes range covered by the observed photometric SED.  
	
	:param wunit: (default: 'micron')
		Wavelength unit. Options are: (1)0 or 'angstrom' for Angstrom unit and (2)1 or 'micron' for micron unit.

	:param funit: (default: 'erg/s/cm2/A')
		Flux unit. Options are: (1)0 or 'erg/s/cm2/A', (2)1 or 'erg/s/cm2', and (3)2 or 'Jy'.

	:param decompose: (default: 1)
		Flag stating whether the best-fit model SED is broken-down into its components (value: 1 or True) or not (value: 0 or False).

	:param shadow_plot: (default: 1)
		Switch for shadow plot (a plot in which uncertainty is plotted as transprent region around the median value). 
		Options are: (1)1 or True and (2)0 or False. If shadow_plot=0 or False, actual model SEDs are plotted.

	:param add_neb_emission: (default: None)
		Flag stating whether to include emission lines. Options are: None, 0, and 1. If None, the decision whether to plot emission lines 
		or not is based on the `add_neb_emission` flag in the header of the `name_sampler_fits` (which tells whether the emission lines modeling
		is included in the fitting).

	:param cosmo: (default: 0)
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0: (default: 70.0)
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0: (default: 0.3)
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.
	
	:param xticks: (optional, default: None)
		List of ticks values in x-axis. If None, the default from matplotlib is used. If xticks is not None, the accepted input is in list format  
		xticks = []. The unit should be the same as the input wunit.

	:param photo_color: (optional, default: 'red')
		Color of photometric fluxes points (in string). The accepted input is the same as that available in matplotlib.

	:param residual_range: (default: [-1.0,1.0])
		Residuals between observed SED and the median posterior model SED. 
		The residual in each band is defined as (f_D - f_M)/f_D, where f_D is flux in observed SED and f_M is flux in model SED.

	:param fontsize_tick: (optional, default: 20)
		Fontsize for the tick. Only relevant if xticks is not None. 
	
	:param fontsize_label: (optional, default: 25)
		Fontsize for the x- and y-axis labels. 

	:param fontsize_legend: (optional, default: 18)
		Fontsize for the legend.

	:param markersize: (optional, default: 100)
		Size of the maarkers associated with the observed and model SEDs.

	:param lw: (optional, default: 1)
		Line width of the model SEDs.
	
	:param name_plot: (optional, default: None)
		Name of the output plot. This is optional parameter.


	:returns name_plot:
		Name of the output pot.

	:returns wave:
		Wavelength grids of the median posterior model SED.

	:returns p16_spec_tot:
		The 16th percentile posterior model spectroscopic SED.

	:returns p50_spec_tot:
		The 50th percentile posterior model spectroscopic SED.

	:returns p84_spec_tot:
		The 84th percentile posterior model spectroscopic SED.

	:returns photo_cwave:
		Central wavelengths of the photometric filters.

	:returns p50_photo_flux:
		Median posterior model photometric SED.

	:returns residuals:
		Residuals.
	"""

	from matplotlib.gridspec import GridSpec

	def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta', 'log_age','dust_index','dust1','dust2',
					'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

	def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
				'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
				'log_tau':0.4,'logzsol':0.0}

	# Open the FITS file
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	data_samplers = hdu[1].data
	hdu.close()
	
	sfh_form = header_samplers['sfh_form']
	dust_ext_law = header_samplers['dust_ext_law']
	duste_switch = header_samplers['duste_stat'] 
	if add_neb_emission == None:
		add_neb_emission = int(header_samplers['add_neb_emission'])
	else:
		add_neb_emission = add_neb_emission

	gas_logu = -2.0
	if add_neb_emission == 1:
		if gas_logu == -2.0:
			if 'gas_logu' in header_samplers: 
				gas_logu = float(header_samplers['gas_logu'])
			else:
				gas_logu = -2.0

	# parameters in the fitting
	nparams = int(header_samplers['nparams'])
	params = []
	for pp in range(0,nparams):
		str_temp = 'param%d' % pp
		params.append(header_samplers[str_temp])

	# filters and observed SED
	nbands = int(header_samplers['nfilters'])
	filters = []
	obs_fluxes = np.zeros(nbands)
	obs_flux_err = np.zeros(nbands)
	for bb in range(0,nbands):
		str_temp = 'fil%d' % bb
		filters.append(header_samplers[str_temp])
		str_temp = "flux%d" % bb
		obs_fluxes[bb] = float(header_samplers[str_temp])
		str_temp = "flux_err%d" % bb
		obs_flux_err[bb] = float(header_samplers[str_temp])

	# central wavelength of all filters
	#photo_cwave = filtering.cwave_filters(filters)
	photo_cwave = cwave_filters(filters)
	
	free_z = int(header_samplers['free_z'])
	nsamplers = int(header_samplers['nrows'])
	imf = int(header_samplers['imf'])
	add_agn = int(header_samplers['add_agn'])

	## get igm_absorption switch:
	add_igm_absorption = int(header_samplers['add_igm_absorption'])
	if add_igm_absorption == 1:
		igm_type = int(header_samplers['igm_type'])
	elif add_igm_absorption == 0:
		igm_type = 0

	# redshift
	if free_z == 0:
		gal_z = float(header_samplers['gal_z'])
		def_params_val['z'] = gal_z

	if duste_switch == 'duste':
		if 'dust_index' in header_samplers:
			def_params_val['dust_index'] = float(header_samplers['dust_index'])

	# cosmology parameter
	if 'cosmo' in header_samplers:
		cosmo = header_samplers['cosmo']
		H0 = float(header_samplers['H0'])
		Om0 = float(header_samplers['Om0'])
	else:
		cosmo = cosmo
		H0 = H0
		Om0 = Om0

	# call FSPS
	global sp
	sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)

	## exclude saturated samplers: log(SFR)~-29.99..
	idx_sel = np.where(data_samplers['log_sfr']>-29.0)

	#rand_idx = np.random.uniform(0,nsamplers,nchains)
	rand_idx = np.random.uniform(0,len(idx_sel[0]),nchains)

	rand_wave = []
	rand_spec_tot = []
	rand_spec_stellar = []
	rand_spec_duste = []
	rand_spec_agn = []
	rand_spec_nebe = []
	rand_photo_flux = []
	for ii in range(0,nchains):
		#idx = rand_idx[ii]
		idx = idx_sel[0][int(rand_idx[ii])]

		params_val = def_params_val
		for pp in range(0,nparams):
			params_val[params[pp]] = data_samplers[params[pp]][int(idx)]

		spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
							add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
							igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,sfh_form=sfh_form,funit=funit)

		if ii == 0:
			if wunit==0 or wunit=='angstrom':
				rand_wave = spec_SED['wave']
			elif wunit==1 or wunit=='micron':
				rand_wave = spec_SED['wave']/1.0e+4

		rand_spec_tot.append(spec_SED['flux_total'])
		rand_spec_stellar.append(spec_SED['flux_stellar'])

		if add_neb_emission == 1:
			rand_spec_nebe.append(spec_SED['flux_nebe'])
		if duste_switch == 1 or duste_switch=='duste':
			rand_spec_duste.append(spec_SED['flux_duste'])
		if add_agn == 1:
			rand_spec_agn.append(spec_SED['flux_agn'])

		# photometric SED
		#mod_fluxes = filtering.filtering(spec_SED['wave'],spec_SED['flux_total'],filters)
		mod_fluxes = filtering(spec_SED['wave'],spec_SED['flux_total'],filters)
		rand_photo_flux.append(mod_fluxes)

	# plotting
	fig1 = plt.figure(figsize=(14,9))
	gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], left=0.1, right=0.96, top=0.98, 
					bottom=0.13, hspace=0.001)

	f1 = fig1.add_subplot(gs[0])
	plt.setp(f1.get_xticklabels(), visible=False)

	if logscale_y == True:
		f1.set_yscale('log')
	if logscale_x == True:
		f1.set_xscale('log')

	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

	if funit=='erg/s/cm2/A' or funit==0:
		plt.ylabel(r'$F_{\lambda}$ [erg $s^{-1}cm^{-2}\AA^{-1}$]', fontsize=int(fontsize_label))
	elif funit=='erg/s/cm2' or funit==1:
		plt.ylabel(r'$\lambda F_{\lambda}$ [erg $s^{-1}cm^{-2}$]', fontsize=int(fontsize_label))
	elif funit=='Jy' or funit==2:
		plt.ylabel(r'$F_{\nu}$ [Jy]', fontsize=int(fontsize_label))
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	# Convert unit of observed SED:
	if funit=='erg/s/cm2/A' or funit==0:
		obs_fluxes = obs_fluxes
		obs_flux_err = obs_flux_err
	elif funit=='erg/s/cm2' or funit==1:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)
	elif funit=='Jy' or funit==2:
		obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
		obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if yrange == None:
		plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
	if yrange != None:
		plt.ylim(yrange[0],yrange[1])

	if xrange == None:
		if wunit==0 or wunit=='angstrom':
			plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
			xmin = min(photo_cwave)*0.7
			xmax = max(photo_cwave)*1.3
		elif wunit==1 or wunit=='micron':
			plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
			xmin = min(photo_cwave)*0.7/1e+4
			xmax = max(photo_cwave)*1.3/1e+4
	elif xrange != None:
		plt.xlim(xrange[0],xrange[1])
		xmin = xrange[0]
		xmax = xrange[1]


	p16_spec_tot = []
	p50_spec_tot = []
	p84_spec_tot = []
	if shadow_plot==1 or shadow_plot==True:
		# plot total:
		p16_spec_tot = np.percentile(rand_spec_tot,16,axis=0)
		p50_spec_tot = np.percentile(rand_spec_tot,50,axis=0)
		p84_spec_tot = np.percentile(rand_spec_tot,84,axis=0)
		f1.fill_between(rand_wave,p16_spec_tot,p84_spec_tot,facecolor='gray',alpha=0.5,zorder=9)
		if decompose==1 or decompose==True:
			plt.plot(rand_wave,p50_spec_tot,lw=lw,color='black',zorder=9,label='total')
		elif decompose==0 or decompose==False:
			plt.plot(rand_wave,p50_spec_tot,lw=lw,color='black',zorder=9)

		if decompose==1 or decompose==True:
			# stellar emission
			f1.fill_between(rand_wave,np.percentile(rand_spec_stellar,16,axis=0),np.percentile(rand_spec_stellar,84,axis=0),facecolor='orange',
									alpha=0.25,zorder=8)
			plt.plot(rand_wave,np.percentile(rand_spec_stellar,50,axis=0),lw=lw,color='darkorange',zorder=8,label='stellar emission')
			# nebular emission
			if add_neb_emission == 1:
				f1.fill_between(rand_wave,np.percentile(rand_spec_nebe,16,axis=0),np.percentile(rand_spec_nebe,84,axis=0),
									facecolor='cyan',alpha=0.25,zorder=8)
				plt.plot(rand_wave,np.percentile(rand_spec_nebe,50,axis=0),lw=lw,color='darkcyan',zorder=8,label='nebular emission')
			# dust emission
			if duste_switch == 1 or duste_switch == 'duste':
				f1.fill_between(rand_wave,np.percentile(rand_spec_duste,16,axis=0),np.percentile(rand_spec_duste,84,axis=0),facecolor='red',
									alpha=0.25,zorder=8)
				plt.plot(rand_wave,np.percentile(rand_spec_duste,50,axis=0),lw=lw,color='darkred',zorder=8,label='dust emission')
			# AGN emission
			if add_agn == 1:
				f1.fill_between(rand_wave,np.percentile(rand_spec_agn,16,axis=0),np.percentile(rand_spec_agn,84,axis=0),facecolor='green',
									alpha=0.25,zorder=8)
				plt.plot(rand_wave,np.percentile(rand_spec_agn,50,axis=0),lw=lw,color='darkgreen',zorder=8,label='AGN torus emission')

			plt.legend(fontsize=int(fontsize_legend), loc=2, ncol=2)

	elif shadow_plot==0 or shadow_plot==False:
		if decompose==1 or decompose==True:
			for ii in range(0,nchains):
				# total
				plt.plot(rand_wave,rand_spec_tot[ii],lw=0.8,color='gray',alpha=0.6)
				# stellar
				plt.plot(rand_wave,rand_spec_stellar[ii],lw=0.5,color='orange',alpha=0.4)
				if add_neb_emission == 1:
					# nebular emission
					plt.plot(rand_wave,rand_spec_nebe[ii],lw=0.5,color='cyan',alpha=0.4)
				if duste_switch == 1 or duste_switch == 'duste':
					# dust emission
					plt.plot(rand_wave,rand_spec_duste[ii],lw=0.5,color='red',alpha=0.4)
				if add_agn == 1:
					# AGN
					plt.plot(rand_wave,rand_spec_agn[ii],lw=0.5,color='green',alpha=0.4)

		elif decompose==0 or decompose==False:
			for ii in range(0,nchains):
				# total
				plt.plot(rand_wave,rand_spec_tot[ii],lw=0.8,color='gray',alpha=0.6)

	p50_photo_flux = np.percentile(rand_photo_flux,50,axis=0)

	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,p50_photo_flux, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9,alpha=1.0)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,p50_photo_flux, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9,alpha=1.0)

	# observed SED
	if wunit==0 or wunit=='angstrom':
		plt.errorbar(photo_cwave,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave,obs_fluxes, s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)
	elif wunit==1 or wunit=='micron':
		plt.errorbar(photo_cwave/1.0e+4,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
		plt.scatter(photo_cwave/1.0e+4,obs_fluxes, s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)

	## plot residual
	f1 = fig1.add_subplot(gs[1])
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

	if logscale_x == True:
		f1.set_xscale('log')

	plt.ylabel(r'residual', fontsize=25)
	plt.ylim(residual_range[0],residual_range[1])
	if wunit==0 or wunit=='angstrom':
		plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
	elif wunit==1 or wunit=='micron':
		plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

	if xticks != None:
		plt.xticks(xticks)
	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	if xrange == None:
		if wunit==0 or wunit=='angstrom':
			plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
			xmin = min(photo_cwave)*0.7
			xmax = max(photo_cwave)*1.3
		elif wunit==1 or wunit=='micron':
			plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
			xmin = min(photo_cwave)*0.7/1e+4
			xmax = max(photo_cwave)*1.3/1e+4
	elif xrange != None:
		plt.xlim(xrange[0],xrange[1])
		xmin = xrange[0]
		xmax = xrange[1]

	# get residual:
	residuals = (obs_fluxes-p50_photo_flux)/obs_fluxes

	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,residuals, s=80, marker='s', lw=3.5, 
							color='gray', zorder=9, alpha=1.0)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,residuals, s=80, marker='s', lw=3.5, 
							color='gray', zorder=9, alpha=1.0)

	x = np.linspace(xmin,xmax,100)
	y = x-x
	plt.plot(x,y,lw=2,color='black',linestyle='--')

	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "sed_%s.png" % (name_sampler_fits1)
		
	plt.subplots_adjust(left=0.25, right=0.98, bottom=0.25, top=0.98)
	plt.savefig(name_plot)

	wave = rand_wave

	return name_plot,wave,p16_spec_tot,p50_spec_tot,p84_spec_tot,photo_cwave,p50_photo_flux,residuals



def plot_corner(name_sampler_fits=None, params=['log_sfr','log_mass','log_dustmass','log_fagn','log_fagn_bol','log_tauagn',
	'log_qpah','log_umin','log_gamma','dust1','dust2','dust_index','log_mw_age','log_age','log_t0','log_alpha','log_beta',
	'log_tau','logzsol','z'], label_params={'log_sfr':'log(SFR[$M_{\odot}yr^{-1}$])','log_mass':'log($M_{*}[M_{\odot}]$)',
	'log_dustmass':'log($M_{dust}$)','log_fagn':'log($f_{AGN,*}$)','log_fagn_bol':'log($f_{AGN,bol}$)',
	'log_tauagn':'log($\\tau_{AGN}$)','log_qpah':'log($Q_{PAH}$)','log_umin':'log($U_{min}$)','log_gamma':'log($\gamma_{e}$)',
	'dust1':'$\hat \\tau_{1}$','dust2':'$\hat \\tau_{2}$', 'dust_index':'$n$', 'log_mw_age':'log($\mathrm{age}_{\mathrm{MW}}$[Gyr])',
	'log_age':'log($\mathrm{age}_{\mathrm{sys}}$[Gyr])','log_t0':'log($t_{0}$[Gyr])','log_alpha':'log($\\alpha$)', 
	'log_beta':'log($\\beta$)','log_tau':'log($\\tau$[Gyr])','logzsol':'log($Z/Z_{\odot}$)','z':'z'}, 
	params_ranges = {'log_sfr':[-99.0,-99.0],'log_mass':[-99.0,-99.0],'log_dustmass':[-99.0,-99.0],'log_fagn':[-5.0,0.48],
	'log_fagn_bol':[-99.0,-99.0],'log_tauagn':[0.70,2.18],'log_qpah':[-1.0, 0.845],'log_umin':[-1.0, 1.176],'log_gamma':[-3.0,-0.824],
	'dust1':[0.0,3.0],'dust2':[0.0, 3.0], 'dust_index':[-2.2,0.4],'log_mw_age':[-99.0,-99.0],'log_age': [-2.5, 1.14],
	'log_t0': [-2.0, 1.14],'log_alpha':[-2.5,2.5],'log_beta':[-2.5,2.5],'log_tau': [-2.5, 1.5], 'logzsol': [-2.0, 0.5], 
	'z': [-99.0, -99.0]}, nbins=12, fontsize_label=20, fontsize_tick=14, name_plot=None):
	
	"""Function for producing corner plot that shows 1D and joint 2D posterior probability distributions from the fitting results with MCMC method.
	
	:param name_sampler_fits: (Mandatory, default: None)
		Name of the input FITS file containing sampler chains from the MCMC fitting. 
		This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.

	:param params: (optional)
		List of parameters that want to be included in the corner plot. This is optional parameter.

	:param label_params: (optional)
		Labels for the parameters in a dictionary format.

	:param params_ranges: (optional)
		Ranges for the parameters to be shown in the plot.

	:param nbins: (default: 12)
		Number of bins to be made in a parameter space when examining the posterior probability function.

	:param fontsize_label: (optional, default: 20)
		Fontsize for the x- and y-axis labels.

	:param fontsize_tick: (optional, default: 14)
		Fontsize for the tick. Only relevant if xticks is not None. 

	:param name_plot: (optional, default: None)
		Desired name for the output plot. 

	:returns name_plot:
		Output plot.
	"""

	def_params=['log_sfr','log_mass','log_dustmass','log_fagn','log_fagn_bol','log_tauagn','log_qpah','log_umin',
				'log_gamma','dust1','dust2','dust_index','log_mw_age','log_age','log_t0','log_alpha','log_beta',
				'log_tau','logzsol','z']

	# open the input FITS file
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	data_samplers = hdu[1].data
	hdu.close()
	# number of parameters
	nparams = len(params)

	if nparams == len(def_params):                        # if default set is used 
		nparams_fit = int(header_samplers['ncols']) - 1   # add SFR, mw_age, log_dustmass
		params_new = []
		label_params_new = {}
		params_ranges_new = {}
		for ii in range(0,nparams):
			for jj in range(0,nparams_fit):
				str_temp = 'col%d' % (jj+2)
				if params[ii] == header_samplers[str_temp]:
					params_new.append(params[ii])
					label_params_new[params[ii]] = label_params[params[ii]]
					params_ranges_new[params[ii]] = params_ranges[params[ii]]
	else:
		params_new = params
		label_params_new = label_params
		params_ranges_new = params_ranges

	## exclude saturated samplers: log(SFR)~-29.99..
	idx_sel = np.where(data_samplers['log_sfr']>-29.0)

	nparams_new = len(params_new)
	nchains = len(idx_sel[0])
	param_samplers = np.zeros((nparams_new,nchains))
	for ii in range(0,nparams_new):
		param_samplers[ii] = [data_samplers[params_new[ii]][j] for j in idx_sel[0]]


	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "corner_%s.png" % (name_sampler_fits1)

	# change the format of label_params, true_params, and postmean_flag:
	label_params1 = []
	params_ranges1 = np.zeros((nparams_new,2))
	for ii in range(0,nparams_new):
		label_params1.append(label_params_new[params_new[ii]])
		params_ranges1[ii][0] = params_ranges_new[params_new[ii]][0]
		params_ranges1[ii][1] = params_ranges_new[params_new[ii]][1]

	plot_triangle_posteriors(param_samplers=param_samplers,label_params=label_params1,params_ranges=params_ranges1,
							nbins=nbins,fontsize_label=fontsize_label,fontsize_tick=fontsize_tick,output_name=name_plot)

	return name_plot


def old_plot_corner(name_sampler_fits=None, params=['log_sfr','log_mass','log_dustmass','log_fagn','log_fagn_bol','log_tauagn','log_qpah','log_umin','log_gamma',
				'dust1','dust2','dust_index','log_mw_age','log_age','log_t0','log_alpha','log_beta','log_tau','logzsol','z'], 
				label_params={'log_sfr':'log(SFR[$M_{\odot}yr^{-1}$])','log_mass':'log($M_{*}[M_{\odot}]$)','log_dustmass':'log($M_{dust}$)','log_fagn':'log($f_{AGN,*}$)',
				'log_fagn_bol':'log($f_{AGN,bol}$)','log_tauagn':'log($\\tau_{AGN}$)','log_qpah':'log($Q_{PAH}$)','log_umin':'log($U_{min}$)','log_gamma':'log($\gamma_{e}$)',
				'dust1':'$\hat \\tau_{1}$','dust2':'$\hat \\tau_{2}$', 'dust_index':'$n$', 'log_mw_age':'log($\mathrm{age}_{\mathrm{MW}}$[Gyr])','log_age':'log($\mathrm{age}_{\mathrm{sys}}$[Gyr])',
				'log_t0':'log($t_{0}$[Gyr])','log_alpha':'log($\\alpha$)', 'log_beta':'log($\\beta$)','log_tau':'log($\\tau$[Gyr])','logzsol':'log($Z/Z_{\odot}$)','z':'z'}, 
				true_params = {'log_sfr':-99.0,'log_mass': -99.0,'log_dustmass':-99.0,'log_fagn':-99.0,'log_fagn_bol':-99.0,'log_tauagn':-99.0,'log_qpah': -99.0,'log_umin': -99.0,'log_gamma': -99.0,
				'dust1':-99.0,'dust2': -99.0,'dust_index':-99.0,'log_mw_age':-99.0,'log_age': -99.0,'log_t0':-99.0, 'log_alpha':-99.0, 'log_beta':-99.0, 'log_tau': -99.0,
				'logzsol': -99.0,'z': -99.0}, postmean_flag = {'log_sfr':0, 'log_mass':0, 'log_dustmass':0, 'log_fagn':0, 'log_fagn_bol':0,'log_tauagn':0,'log_qpah':0,'log_umin':0,
				'log_gamma':0,'dust1':0,'dust2':0,'dust_index':0,'log_mw_age':0,'log_age':0,'log_t0':0,'log_alpha':0,'log_beta':0,'log_tau':0,'logzsol':0,'z':0}, 
				postmode_flag = {'log_sfr':0, 'log_mass':0, 'log_dustmass':0,'log_fagn':0,'log_fagn_bol':0,'log_tauagn':0,'log_qpah':0,'log_umin':0,'log_gamma':0,'dust1':0,
				'dust2':0,'dust_index':0,'log_mw_age':0,'log_age':0,'log_t0':0,'log_alpha':0,'log_beta':0,'log_tau':0,'logzsol':0,'z':0}, 
				params_ranges = {'log_sfr':[-99.0,-99.0],'log_mass':[-99.0,-99.0],'log_dustmass':[-99.0,-99.0],'log_fagn':[-5.0,0.48],'log_fagn_bol':[-99.0,-99.0],'log_tauagn':[0.70,2.18],
				'log_qpah':[-1.0, 0.845],'log_umin':[-1.0, 1.176],'log_gamma':[-3.0,-0.824],'dust1':[0.0,3.0],'dust2':[0.0, 3.0], 'dust_index':[-2.2,0.4],
				'log_mw_age':[-99.0,-99.0],'log_age': [-2.5, 1.14],'log_t0': [-2.0, 1.14],'log_alpha':[-2.5,2.5],'log_beta':[-2.5,2.5],'log_tau': [-2.5, 1.5], 
				'logzsol': [-2.0, 0.5], 'z': [-99.0, -99.0]}, nbins=12, fontsize_label=20, fontsize_tick=14, name_plot=None):
	
	"""Function for producing corner plot that shows 1D and joint 2D posterior probability distributions from the fitting results with MCMC method.
	
	:param name_sampler_fits: (Mandatory, default: None)
		Name of the input FITS file containing sampler chains from the MCMC fitting. 
		This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.

	:param params: (optional)
		List of parameters that want to be included in the corner plot. This is optional parameter.

	:param label_params: (optional)
		Labels for the parameters in a dictionary format.

	:param true_params: (optional)
		True values of the parameters in case exist and want to be displayed in the corner plot.

	:param postmean_flag: (optional)
		Flag stating whether to inlude (value: 1) values of meaan poterior in the corner plot (other than the median posterior) or not (value: 0).

	:param postmode_flag: (optional)
		Flag stating whether to include (value: 1) values of mode posterior in the corner plot (other than the median posterior) or not (value: 0).

	:param params_ranges: (optional)
		Ranges for the parameters to be shown in the plot.

	:param nbins: (default: 12)
		Number of bins to be made in a parameter space when examining the posterior probability function.

	:param fontsize_label: (optional, default: 20)
		Fontsize for the x- and y-axis labels.

	:param fontsize_tick: (optional, default: 14)
		Fontsize for the tick. Only relevant if xticks is not None. 

	:param name_plot: (optional, default: None)
		Desired name for the output plot. 

	:returns name_plot:
		Output plot.
	"""

	# open the input FITS file
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	data_samplers = hdu[1].data
	hdu.close()
	# number of parameters
	nparams = len(params)

	if nparams == 20:                                     # if default set is used 
		nparams_fit = int(header_samplers['ncols']) - 1   # add SFR, mw_age, log_dustmass
		params_new = []
		label_params_new = {}
		true_params_new = {}
		postmean_flag_new = {}
		postmode_flag_new = {}
		params_ranges_new = {}
		for ii in range(0,nparams):
			for jj in range(0,nparams_fit):
				str_temp = 'col%d' % (jj+2)
				if params[ii] == header_samplers[str_temp]:
					params_new.append(params[ii])
					label_params_new[params[ii]] = label_params[params[ii]]
					postmean_flag_new[params[ii]] = postmean_flag[params[ii]]
					postmode_flag_new[params[ii]] = postmode_flag[params[ii]]
					params_ranges_new[params[ii]] = params_ranges[params[ii]]
					true_params_new[params[ii]] = true_params[params[ii]]
	else:
		params_new = params
		label_params_new = label_params
		postmean_flag_new = postmean_flag
		postmode_flag_new = postmode_flag
		params_ranges_new = params_ranges
		true_params_new = {}
		for ii in range(0,nparams):
			true_params_new[params_new[ii]] = true_params[params_new[ii]]

	#nparams_new = len(params_new)
	#nchains = header_samplers['nrows']
	#param_samplers = np.zeros((nparams_new,nchains))
	#for ii in range(0,nparams_new):
	#	param_samplers[ii] = data_samplers[params_new[ii]]


	## exclude saturated samplers: log(SFR)~-29.99..
	idx_sel = np.where(data_samplers['log_sfr']>-29.0)

	nparams_new = len(params_new)
	nchains = len(idx_sel[0])
	param_samplers = np.zeros((nparams_new,nchains))
	for ii in range(0,nparams_new):
		param_samplers[ii] = [data_samplers[params_new[ii]][j] for j in idx_sel[0]]


	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "corner_%s.png" % (name_sampler_fits1)

	# change the format of label_params, true_params, and postmean_flag:
	label_params1 = []
	true_params1 = np.zeros(nparams_new)
	postmean_flag1 = np.zeros(nparams_new)
	postmode_flag1 = np.zeros(nparams_new)
	params_ranges1 = np.zeros((nparams_new,2))
	for ii in range(0,nparams_new):
		label_params1.append(label_params_new[params_new[ii]])
		true_params1[ii] = true_params_new[params_new[ii]]
		postmean_flag1[ii] = postmean_flag_new[params_new[ii]]
		postmode_flag1[ii] = postmode_flag_new[params_new[ii]]
		params_ranges1[ii][0] = params_ranges_new[params_new[ii]][0]
		params_ranges1[ii][1] = params_ranges_new[params_new[ii]][1]

	plot_triangle_posteriors(param_samplers=param_samplers,label_params=label_params1,true_params=true_params1,
										post_mean_flag=postmean_flag1,post_mode_flag=postmode_flag1,params_ranges=params_ranges1,
										nbins=nbins,fontsize_label=fontsize_label,fontsize_tick=fontsize_tick,output_name=name_plot)

	return name_plot


def plot_sfh_mcmc(name_sampler_fits=None, nchains=200, del_t=0.05, plot_true=0, true_params = {'log_tau': -99.0, 'log_age': -99.0, 
	'log_t0': -99.0, 'log_alpha':-99.0, 'log_beta':-99.0, 'log_mass': -99.0}, true_SFH_lbt=[], true_SFH_sfr=[],lbacktime_max=None, 
	yrange=None, loc_legend=2, fontsize_tick=18, fontsize_label=25, fontsize_legend=26, logscale_x=False, logscale_y=False, name_plot=None):
	"""Function for producing SFH plot from fitting result obtained with the MCMC method.

	:param name_sampler_fits: (Mandatory, default: None)
		Name of the input FITS file containing sampler chains from the MCMC fitting. 
		This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.

	:param nchains: (default: 200)
		Number of randomly-chosen sampler chains to be used for calculating the inferred SFH.

	:param del_t: (default: 0.05)
		Increment in the look-back time (in Gyr) to be used in sampling the SFH.

	:param plot_true: (default: 0)
		A flag stating whether to plot (value:1) the true SFH or not (value:0).

	:param true_params: (optional)
		True values of the true parameters if available and the true SFH is intended to be shown. Only relevant if plot_true=1.
		This should be in dictionary format as shown in the default set.

	:param true_SFH_lbt: (default: [])
		1D array of the true (arbitrary) SFH -the time look-back time component. In case the true SFH is not represented by parametric form.

	:param true_SFH_sfr: (default: [])
		1D arrays of the true (arbitrary) SFH -the SFR component. In case the true SFH is not represented by parametric form.

	:param lbacktime_max: (optional, default: None)
		Maximum look-back time in the SFH plot. If None, the maximum look-back time is defined from the age of universe at the redshift of the galaxy.

	:param yrange: (optional, default: None)
		Range in the y-axis.

	:param loc_legend: (optional, default: 2)
		Where to locate the legend. This is the same numbering as in the `matplotlib`.

	:param fontsize_tick: (optional, default: 18)
		Fontsize for the tick. Only relevant if xticks is not None. 
	
	:param fontsize_label: (optional, default: 25)
		Fontsize for the x- and y-axis labels. 

	:param fontsize_legend: (optional, default: 26)
		Fontsize for the legend.

	:param logscale_x: (optional, default: True)
		Flag stating whether the x-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param logscale_y: (optional, default: True)
		Flag stating whether the y-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param name_plot:
		Desired name for the output plot.


	:returns name_plot:
		Name of the output plot.

	:returns grid_lbt:
		Look-back time grids in the SFH.

	:returns grid_sfr_p16:
		16th percentile of the SFH.

	:returns grid_sfr_p50:
		Median of the SFH.

	:returns grid_sfr_p84:
		84th percentile of the SFH.
	"""
	
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	data_samplers = hdu[1].data
	hdu.close()

	sfh_form = header_samplers['sfh_form']
	nsamplers = len(data_samplers['log_age'])
	gal_z = float(header_samplers['gal_z'])

	# cosmology parameter
	cosmo = header_samplers['cosmo']
	H0 = float(header_samplers['H0'])
	Om0 = float(header_samplers['Om0'])

	if cosmo == 'flat_LCDM':
		cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
		max_lbt = cosmo1.age(gal_z).value
	elif cosmo == 'WMAP5':
		max_lbt = WMAP5.age(gal_z).value
	elif cosmo == 'WMAP7':
		max_lbt = WMAP7.age(gal_z).value
	elif cosmo == 'WMAP9':
		max_lbt = WMAP9.age(gal_z).value
	elif cosmo == 'Planck13':
		max_lbt = Planck13.age(gal_z).value
	elif cosmo == 'Planck15':
		max_lbt = Planck15.age(gal_z).value
	#elif cosmo == 'Planck18':
	#	max_lbt = Planck18.age(gal_z).value

	nt = int(max_lbt/del_t)
	grid_lbt = np.linspace(0.0,max_lbt,nt)
	array_sfr_at_lbt = np.zeros((nchains,nt))

	## exclude saturated samplers: log(SFR)~-29.99..
	idx_sel = np.where(data_samplers['log_sfr']>-29.0)

	#rand_idx = np.random.uniform(0,nsamplers,nchains)
	rand_idx = np.random.uniform(0,len(idx_sel[0]),nchains)

	for ii in range(0,nchains):
		#idx = random.randint(0,nsamplers-1)
		idx = idx_sel[0][int(rand_idx[ii])]

		age = math.pow(10.0,data_samplers['log_age'][idx])
		tau = math.pow(10.0,data_samplers['log_tau'][idx])
		t0 = 0.0
		alpha = 0.0
		beta = 0.0
		if sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh':
			t0 = math.pow(10.0,data_samplers['log_t0'][idx])
		if sfh_form=='double_power_sfh':
			alpha = math.pow(10.0,data_samplers['log_alpha'][idx])
			beta = math.pow(10.0,data_samplers['log_beta'][idx])

		formed_mass = math.pow(10.0,data_samplers['log_mass'][idx])

		t,SFR_t0 = construct_SFH(sfh_form=sfh_form,t0=t0,tau=tau,alpha=alpha,beta=beta,age=age,formed_mass=formed_mass)
		t_back0 = np.abs(t - age)

		t_back = np.zeros(1) + max_lbt
		t_back = np.append(t_back,t_back0)
		SFR_t = np.zeros(1)
		SFR_t = np.append(SFR_t,SFR_t0)

		f = interp1d(t_back,SFR_t,fill_value="extrapolate")
		array_sfr_at_lbt[ii] = f(grid_lbt)

	array_sfr_at_lbt_trans = np.transpose(array_sfr_at_lbt, axes=(1,0))
	grid_sfr_p16 = np.percentile(array_sfr_at_lbt_trans, 16, axis=1)
	grid_sfr_p50 = np.percentile(array_sfr_at_lbt_trans, 50, axis=1)
	grid_sfr_p84 = np.percentile(array_sfr_at_lbt_trans, 84, axis=1)

	# plotting
	fig = plt.figure(figsize=(8,5))
	f1 = plt.subplot()
	if logscale_y == True:
		f1.set_yscale('log')
	if logscale_x == True:
		f1.set_xscale('log')
	plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	plt.tick_params(axis='y', which='both', right='on')
	plt.tick_params(axis='x', which='both', top='on')
	plt.xlabel('Look back time [Gyr]', fontsize=int(fontsize_label))
	plt.ylabel(r'SFR[$M_{\odot}yr^{-1}$]', fontsize=int(fontsize_label))

	f1.fill_between(grid_lbt, grid_sfr_p16, grid_sfr_p84, color='gray', alpha=0.5)
	plt.plot(grid_lbt,grid_sfr_p50,lw=4,color='black')

	# xrange:
	if lbacktime_max == None:
		xmax = max_lbt
		plt.xlim(xmax,0)
	elif lbacktime_max != None:
		xmax = lbacktime_max
		plt.xlim(xmax,0)

	# yrange:
	if yrange == None:
		maxSFR = max(grid_sfr_p84)
		plt.ylim(0,maxSFR*1.2)
	if yrange != None:
		plt.ylim(yrange[0],yrange[1])

	# plot true sfh if required:
	if plot_true==1 or plot_true==True:
		if len(true_SFH_lbt) == 0:
			true_tau = math.pow(10.0,true_params['log_tau'])
			true_age = math.pow(10.0,true_params['log_age'])
			true_mass = math.pow(10.0,true_params['log_mass'])
			true_t0 = 0
			true_alpha = 0
			true_beta = 0
			if sfh_form=='log_normal_sfh' or sfh_form=='gaussian_sfh':
				true_t0 = math.pow(10.0,true_params['log_t0'])
			if sfh_form=='double_power_sfh':
				true_alpha = math.pow(10.0,true_params['log_alpha'])
				true_beta = math.pow(10.0,true_params['log_beta'])

			t,SFR_t = construct_SFH(sfh_form=sfh_form,t0=true_t0,tau=true_tau,alpha=true_alpha,
														beta=true_beta,age=true_age,formed_mass=true_mass)

			t_back = np.abs(t-true_age)
		else:
			t_back = true_SFH_lbt
			SFR_t = true_SFH_sfr

		plt.plot(t_back,SFR_t,lw=4,color='red',zorder=11, label='true')

		plt.legend(fontsize=int(fontsize_legend), loc=loc_legend)

	plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "sfh_%s.png" % (name_sampler_fits1)
	
	plt.savefig(name_plot)

	return name_plot,grid_lbt,grid_sfr_p16,grid_sfr_p50,grid_sfr_p84



def plot_SED_rdsps_save_PDF(name_sampler_fits=[], logscale_x=True, logscale_y=True, xrange=None, yrange=None, wunit='micron', funit='erg/s/cm2/A', 
	decompose=1, plot_true=0, true_params = {'log_sfr': -99.0,'log_mass': -99.0,'log_dustmass':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,
	'log_qpah': -99.0,'log_umin': -99.0,'log_gamma': -99.0,'dust1':-99.0,'dust2': -99.0, 'dust_index':-99.0,'log_mw_age':-99.0,'log_age': -99.0, 
	'log_alpha':-99.0,'log_beta':-99.0, 'log_t0': -99.0, 'log_tau': -99.0,'logzsol': -99.0,'z': -99.0}, xticks=None, photo_color='red', residual_range=[-1.0,1.0],
	fontsize_tick=18, fontsize_label=25, fontsize_legend=18, name_out_PDF=None):

	"""Function for producing an SED plot from a fitting result obtained with the RDSPS method. The output plot inludes residuals between the observed SED and best-fit model SED.  
	In this case, the best-fit model SED in the plot is the one with lowest chi-square from the input set of pre-calculated model SEDs in the fitting. 

	:param name_sampler_fits: (Mandatory, default: None)
		Name of input FITS file containing model SEDs and their probabilities. This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.

	:param logscale_x: (optional, default: True)
		Flag stating whether the x-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param logscale_y: (optional, default: True)
		Flag stating whether the y-axis is plotted in logarithmic scale (value: True) or not (value: False).

	:param xrange: (optional, default: None)
		Range in x-axis. The accepted format is: [xmin,xmax]. If xrange=None, the range will be defined based on 
		the wavelength range covered by the observed photometric SED.

	:param yrange: (optional, default: None)
		Range in y-axis. The accepted format is: [ymin,ymax]. If yrange=None, the range will be defined based on
		the fluxes range covered by the observed photometric SED.  
	
	:param wunit: (default: 'micron')
		Wavelength unit. Options are: (1)0 or 'angstrom' for Angstrom unit and (2)1 or 'micron' for micron unit.

	:param funit: (default: 'erg/s/cm2/A')
		Flux unit. Options are: (1)0 or 'erg/s/cm2/A', (2)1 or 'erg/s/cm2', and (3)2 or 'Jy'.

	:param decompose: (default: 1)
		Flag stating whether the best-fit model SED is broken-down into its components (value: 1 or True) or not (value: 0 or False).

	:param plot_true: (optional, default: 0)
		Flag stating whether to plot true model SED (in case available) or not. Options are: (1)0 or False and (2)1 or True.

	:param true_params: (optional)
		True values of parameters in case available. It should be in a dictionary format as shown in the default set. Only releavant if plot_true=1.

	:param xticks: (optional, default: None)
		List of ticks values in x-axis. If None, the default from matplotlib is used. If xticks is not None, the accepted input is in list format  
		xticks = []. The unit should be the same as the input wunit.

	:param photo_color: (optional, default: 'red')
		Color of photometric fluxes points (in string). The accepted input is the same as that available in matplotlib.

	:param residual_range: (default: [-1.0,1.0])
		Residuals between observed SED and the median posterior model SED. 
		The residual in each band is defined as (f_D - f_M)/f_D, where f_D is flux in observed SED and f_M is flux in model SED.

	:param fontsize_tick: (optional, default: 20)
		Fontsize for the tick. Only relevant if xticks is not None. 
	
	:param fontsize_label: (optional, default: 25)
		Fontsize for the x- and y-axis labels. 

	:param fontsize_legend: (optional, default: 18)
		Fontsize for the legend.
	
	:param name_out_PDF: (optional, default: None)
		Name of the output PDF file. This is optional parameter.
	"""

	from matplotlib.gridspec import GridSpec
	from matplotlib.backends.backend_pdf import PdfPages

	def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta', 'log_age','dust_index','dust1','dust2',
					'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']
	def_params_val={'log_mass':0.0,'z':-99.0,'log_fagn':-99.0,'log_tauagn':-99.0,'log_qpah':-99.0,'log_umin':-99.0,'log_gamma':-99.0,
					'dust1':-99.0,'dust2':-99.0, 'dust_index':-99.0,'log_age':-99.0,'log_alpha':-99.0,'log_beta':-99.0,
					'log_t0':-99.0,'log_tau':-99.0,'logzsol':-99.0}

	if name_out_PDF == None:
		name_out_PDF = "PDF_SED_plots_rdsps.pdf"
	with PdfPages(name_out_PDF) as pdf:
		for zz in range(0,len(name_sampler_fits)):

			# open the FITS file:
			hdu = fits.open(name_sampler_fits[zz])
			header_samplers = hdu[0].header
			data_samplers = hdu[1].data
			hdu.close()

			# some parameters 
			imf = int(header_samplers['imf'])
			sfh_form = header_samplers['sfh_form']
			dust_ext_law = header_samplers['dust_ext_law']
			duste_switch = header_samplers['duste_stat']
			add_neb_emission = int(header_samplers['add_neb_emission'])
			add_agn = header_samplers['add_agn']
			add_igm_absorption = header_samplers['add_igm_absorption']
			if add_igm_absorption == 1:
				igm_type = int(header_samplers['igm_type'])
			elif add_igm_absorption == 0:
				igm_type = 0

			# redshift
			free_z = int(header_samplers['free_z'])
			if free_z == 0:
				gal_z = float(header_samplers['gal_z'])
				def_params_val['z'] = gal_z

			# cosmology parameter
			cosmo = header_samplers['cosmo']
			H0 = float(header_samplers['H0'])
			Om0 = float(header_samplers['Om0'])

			# filters and observed SED
			nbands = int(header_samplers['nfilters'])
			filters = []
			obs_fluxes = np.zeros(nbands)
			obs_flux_err = np.zeros(nbands)
			for bb in range(0,nbands):
				str_temp = 'fil%d' % bb
				filters.append(header_samplers[str_temp])
				str_temp = 'flux%d' % bb
				obs_fluxes[bb] = float(header_samplers[str_temp])
				str_temp = 'flux_err%d' % bb
				obs_flux_err[bb] = float(header_samplers[str_temp])

			# central wavelength of all filters
			photo_cwave = cwave_filters(filters)

			# get list parameters
			nparams0 = int(header_samplers['nparams'])
			params = []
			for pp in range(0,nparams0):
				str_temp = 'param%d' % pp
				params.append(header_samplers[str_temp])
			params.append('log_mass')
			nparams = nparams0 + 1

			# get best-fit parameters
			idx, min_val = min(enumerate(data_samplers['chi2']), key=operator.itemgetter(1))
			bfit_chi2 = data_samplers['chi2'][idx]
			bfit_params = {}
			for pp in range(0,nparams):
				bfit_params[params[pp]] = data_samplers[params[pp]][idx]

			# call fsps
			global sp
			sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf)

			# generate the spectrum
			params_val = def_params_val
			for pp in range(0,nparams):
				params_val[params[pp]] = bfit_params[params[pp]]

			spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
									add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
									igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,sfh_form=sfh_form,funit=funit)

			# get the photometric SED:
			bfit_photo_SED = filtering(spec_SED['wave'],spec_SED['flux_total'],filters)

			if wunit==0 or wunit=='angstrom':
				spec_wave = spec_SED['wave']
			elif wunit==1 or wunit=='micron':
				spec_wave = spec_SED['wave']/1.0e+4


			# plotting
			fig1 = plt.figure(figsize=(14,7))

			gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], left=0.1, right=0.98, top=0.98, 
							bottom=0.13, hspace=0.001)

			f1 = fig1.add_subplot(gs[0])
			plt.setp(f1.get_xticklabels(), visible=False)

			if logscale_y == True:
				f1.set_yscale('log')
			#if logscale_x == True:
			#	f1.set_xscale('log')
			plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
			#plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

			if wunit==0 or wunit=='angstrom':
				plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
			elif wunit==1 or wunit=='micron':
				plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

			if funit=='erg/s/cm2/A' or funit==0:
				plt.ylabel(r'$F_{\lambda}$ [erg $s^{-1}cm^{-2}\AA^{-1}$]', fontsize=int(fontsize_label))
			elif funit=='erg/s/cm2' or funit==1:
				plt.ylabel(r'$\lambda F_{\lambda}$ [erg $s^{-1}cm^{-2}$]', fontsize=int(fontsize_label))
			elif funit=='Jy' or funit==2:
				plt.ylabel(r'$F_{\nu}$ [Jy]', fontsize=int(fontsize_label))
			else:
				print ("The input funit is not recognized!")
				sys.exit()

			if xticks != None:
				plt.xticks(xticks)

			for axis in [f1.xaxis]:
				axis.set_major_formatter(ScalarFormatter())
			if xrange == None:
				if wunit==0 or wunit=='angstrom':
					plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
				elif wunit==1 or wunit=='micron':
					plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
			elif xrange != None:
				plt.xlim(xrange[0],xrange[1])

			# Convert unit of observed SED:
			if funit=='erg/s/cm2/A' or funit==0:
				obs_fluxes = obs_fluxes
				obs_flux_err = obs_flux_err
			elif funit=='erg/s/cm2' or funit==1:
				obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)
				obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)
			elif funit=='Jy' or funit==2:
				obs_fluxes = np.asarray(obs_fluxes)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
				obs_flux_err = np.asarray(obs_flux_err)*np.asarray(photo_cwave)*np.asarray(photo_cwave)/1.0e-23/2.998e+18
			else:
				print ("The input funit is not recognized!")
				sys.exit()

			if yrange == None:
				plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
			if yrange != None:
				plt.ylim(yrange[0],yrange[1])


			# alocate arrays:
			spec_total = spec_SED['flux_total']
			spec_stellar = []
			spec_nebe = []
			spec_duste = []
			spec_agn = []

			if decompose==1 or decompose==True:
				spec_stellar = spec_SED['flux_stellar']
				spec_nebe = spec_SED['flux_nebe']
				spec_duste = spec_SED['flux_duste']
				spec_agn = spec_SED['flux_agn']

				# stellar emission
				plt.plot(spec_wave,spec_stellar,lw=3,color='darkorange',label='stellar emission')
				if add_neb_emission == 1:
					# nebular emission
					plt.plot(spec_wave,spec_nebe,lw=3,color='darkcyan',label='nebular emission')
				if duste_switch == 1 or duste_switch == 'duste':
					# dust emission
					plt.plot(spec_wave,spec_duste,lw=3,color='darkred',label='dust emission')
				if add_agn == 1:
					# AGN dusty torus emission
					plt.plot(spec_wave,spec_agn,lw=3,color='darkgreen',label='AGN torus emission')

				# total
				plt.plot(spec_wave,spec_total,lw=3,color='black',zorder=11,label='total')

				plt.legend(fontsize=int(fontsize_legend), loc=2, ncol=2)

			elif decompose==0 or decompose==False:
				# total
				plt.plot(spec_wave,spec_total,lw=3,color='black',zorder=9)

			# plot true SED if required:
			if plot_true==1 or plot_true==True:
				params_val = def_params_val
				for pp in range(0,nparams):
					params_val[params[pp]] = true_params[params[pp]]

				spec_SED = generate_modelSED_spec_decompose(sp=sp,params_val=params_val, imf=imf, duste_switch=duste_switch,
									add_neb_emission=add_neb_emission,dust_ext_law=dust_ext_law,add_agn=add_agn,add_igm_absorption=add_igm_absorption,
									igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,sfh_form=sfh_form,funit=funit)

				if wunit==0 or wunit=='angstrom':
					wave0 = spec_SED['wave']
				elif wunit==1 or wunit=='micron':
					wave0 = spec_SED['wave']/1.0e+4

				if decompose==1 or decompose==True:
					# stellar emission
					plt.plot(wave0,spec_SED['flux_stellar'],lw=3.0,color='darkorange',linestyle='--')
					# nebular emission
					plt.plot(wave0,spec_SED['flux_nebe'],lw=3.0,color='darkcyan',linestyle='--')
					# dust emission
					plt.plot(wave0,spec_SED['flux_duste'],lw=3.0,color='darkred',linestyle='--')
					# AGN dusty torus emission
					plt.plot(wave0,spec_SED['flux_agn'],lw=3.0,color='darkgreen',linestyle='--')

				# total:
				plt.plot(wave0,spec_SED['flux_total'],lw=3.0,color='black',linestyle='--',zorder=10)


			# plot observed SED
			if wunit==0 or wunit=='angstrom':
				plt.scatter(photo_cwave,bfit_photo_SED, s=250, marker='s', lw=3, edgecolor='gray', color='none', zorder=9)

				plt.errorbar(photo_cwave,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
				plt.scatter(photo_cwave,obs_fluxes, s=250, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)
			elif wunit==1 or wunit=='micron':
				plt.scatter(photo_cwave/1.0e+4,bfit_photo_SED, s=250, marker='s', lw=3, edgecolor='gray', color='none', zorder=9)

				plt.errorbar(photo_cwave/1.0e+4,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o',zorder=10)
				plt.scatter(photo_cwave/1.0e+4,obs_fluxes, s=250, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=11)

			f1.text(0.25, 0.9, "reduced $\chi^2 = %.3f$" % (bfit_chi2/nbands),verticalalignment='bottom', horizontalalignment='right',
			       transform=f1.transAxes,color='black', fontsize=20)

			# plot residual
			f1 = fig1.add_subplot(gs[1])
			plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
			plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

			if logscale_x == True:
				f1.set_xscale('log')

			plt.ylabel(r'residual', fontsize=25)
			plt.ylim(residual_range[0],residual_range[1])
			if wunit==0 or wunit=='angstrom':
				plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
			elif wunit==1 or wunit=='micron':
				plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

			if xticks != None:
				plt.xticks(xticks)
			for axis in [f1.xaxis]:
				axis.set_major_formatter(ScalarFormatter())

			if xrange == None:
				if wunit==0 or wunit=='angstrom':
					plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
					xmin = min(photo_cwave)*0.7
					xmax = max(photo_cwave)*1.3
				elif wunit==1 or wunit=='micron':
					plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
					xmin = min(photo_cwave)*0.7/1e+4
					xmax = max(photo_cwave)*1.3/1e+4
			elif xrange != None:
				plt.xlim(xrange[0],xrange[1])
				xmin = xrange[0]
				xmax = xrange[1]

			# get residual:
			residuals = (obs_fluxes-bfit_photo_SED)/obs_fluxes

			if wunit==0 or wunit=='angstrom':
				plt.scatter(photo_cwave,residuals, s=80, marker='s', lw=3.5, 
									color='gray', zorder=9, alpha=1.0)
			elif wunit==1 or wunit=='micron':
				plt.scatter(photo_cwave/1.0e+4,residuals, s=80, marker='s', lw=3.5, 
									color='gray', zorder=9, alpha=1.0)

			x = np.linspace(xmin,xmax,100)
			y = x-x
			plt.plot(x,y,lw=2,color='black',linestyle='--')	

			#plt.subplots_adjust(left=0.25, right=0.98, bottom=0.25, top=0.98)

			pdf.savefig()
			plt.close()






