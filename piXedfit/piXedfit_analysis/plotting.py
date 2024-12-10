import numpy as np
from math import pow
import sys, os
from operator import itemgetter
import matplotlib
matplotlib.use('agg')
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.cosmology import *

from ..piXedfit_model.model_utils import construct_SFH, convert_unit_spec_from_ergscm2A, list_default_params_fit, default_params_val, get_no_nebem_wave_fit
from ..piXedfit_model.gen_models import generate_modelSED_spec_decompose
from ..utils.filtering import cwave_filters, filtering
from ..utils.posteriors import plot_triangle_posteriors


__all__ = ["plot_SED", "plot_corner", "plot_sfh_mcmc"]


def plot_SED_rdsps_photo(filters=None,obs_photo=None,bfit_photo=None,bfit_mod_spec=None,minchi2_params=None,header_samplers=None,
	logscale_x=True,logscale_y=True,xrange=None,yrange=None,wunit='micron',funit='erg/s/cm2/A',decompose=0,xticks=None,
	photo_color='red',residual_range=[-1.0,1.0],fontsize_tick=18,fontsize_label=25,show_legend=True,loc_legend=4,
	fontsize_legend=18,markersize=100,lw=2.0,name_plot=None):

	from matplotlib.gridspec import GridSpec

	# observed SEDs
	nbands = len(filters)
	obs_fluxes = obs_photo['flux']
	obs_flux_err = obs_photo['flux_err']

	# central wavelength of all filters
	photo_cwave = cwave_filters(filters)

	# convert flux unit of observed SED
	obs_fluxes = convert_unit_spec_from_ergscm2A(photo_cwave,obs_fluxes,funit=funit)
	obs_flux_err = convert_unit_spec_from_ergscm2A(photo_cwave,obs_flux_err,funit=funit)

	# plotting
	fig1 = plt.figure(figsize=(14,7))

	gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], left=0.1, right=0.98, top=0.98, bottom=0.13, hspace=0.001)

	f1 = fig1.add_subplot(gs[0])
	plt.setp(f1.get_xticklabels(), visible=False)

	if logscale_y == True:
		f1.set_yscale('log')
	if logscale_x == True:
		f1.set_xscale('log')
	
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

	#if xticks is not None:
	#	plt.xticks(xticks)
	#for axis in [f1.xaxis]:
	#	axis.set_major_formatter(ScalarFormatter())

	if xrange is None:
		xmin, xmax = min(photo_cwave)*0.7, max(photo_cwave)*1.3
	elif xrange is not None:
		xmin, xmax = xrange[0], xrange[1]

	if wunit==0 or wunit=='angstrom':
		plt.xlim(xmin,xmax)
	elif wunit==1 or wunit=='micron':
		xmin, xmax = xmin/1e+4, xmax/1e+4
		plt.xlim(xmin,xmax)

	if yrange is None:
		plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
	elif yrange is not None:
		plt.ylim(yrange[0],yrange[1])

	if decompose==1 or decompose==True:
		def_params = list_default_params_fit()
		def_params_val = default_params_val()

		# modeling configuration 
		imf = header_samplers['imf']
		sfh_form = header_samplers['sfh_form']
		dust_law = header_samplers['dust_law']
		duste_switch = header_samplers['duste_stat']
		add_neb_emission = header_samplers['add_neb_emission']
		add_agn = header_samplers['add_agn']
		add_igm_absorption = header_samplers['add_igm_absorption']
		if add_igm_absorption == 1:
			igm_type = header_samplers['igm_type']
		elif add_igm_absorption == 0:
			igm_type = 0

		smooth_velocity = header_samplers['smooth_velocity']
		sigma_smooth = header_samplers['sigma_smooth']
		smooth_lsf = header_samplers['smooth_lsf']
		if smooth_lsf == 1 or smooth_lsf == True:
			name_file_lsf = header_samplers['name_file_lsf']
			data = np.loadtxt(temp_dir+name_file_lsf)
			lsf_wave, lsf_sigma = data[:,0], data[:,1]
		elif smooth_lsf == 0:
			lsf_wave, lsf_sigma = None, None

		if header_samplers['free_z'] == 0:
			def_params_val['z'] = float(header_samplers['gal_z'])

		# cosmology parameter
		cosmo = header_samplers['cosmo']
		H0 = float(header_samplers['H0'])
		Om0 = float(header_samplers['Om0'])

		params_val = def_params_val
		for pp in range(0,int(header_samplers['nparams'])):
			str_temp = 'param%d' % pp
			if header_samplers[str_temp] in def_params:
				params_val[header_samplers[str_temp]] = minchi2_params[header_samplers[str_temp]][0]

		spec_SED = generate_modelSED_spec_decompose(params_val=params_val,imf=imf,duste_switch=duste_switch,
										add_neb_emission=add_neb_emission,dust_law=dust_law,add_agn=add_agn,
										add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,
										H0=H0,Om0=Om0,sfh_form=sfh_form,funit=funit,smooth_velocity=smooth_velocity,
										sigma_smooth=sigma_smooth,smooth_lsf=smooth_lsf,lsf_wave=lsf_wave,lsf_sigma=lsf_sigma)

		bfit_photo_fluxes = filtering(spec_SED['wave'],spec_SED['flux_total'],filters)
		bfit_spec_wave = spec_SED['wave']
		bfit_spec_flux_tot = spec_SED['flux_total']

		if wunit==0 or wunit=='angstrom':
			plt.plot(spec_SED['wave'],spec_SED['flux_stellar'],lw=lw,color='darkorange',label='stellar emission', zorder=1)
			if add_neb_emission == 1:
				plt.plot(spec_SED['wave'],spec_SED['flux_nebe'],lw=lw,color='darkcyan',label='nebular emission', zorder=2)
			if duste_switch==1:
				plt.plot(spec_SED['wave'],spec_SED['flux_duste'],lw=lw,color='darkred',label='dust emission', zorder=3)
			if add_agn == 1:
				plt.plot(spec_SED['wave'],spec_SED['flux_agn'],lw=lw,color='darkgreen',label='AGN torus emission', zorder=4)

		elif wunit==1 or wunit=='micron':
			plt.plot(spec_SED['wave']/1.0e+4,spec_SED['flux_stellar'],lw=lw,color='darkorange',label='stellar emission', zorder=1)
			if add_neb_emission == 1:
				plt.plot(spec_SED['wave']/1.0e+4,spec_SED['flux_nebe'],lw=lw,color='darkcyan',label='nebular emission', zorder=2)
			if duste_switch==1:
				plt.plot(spec_SED['wave']/1.0e+4,spec_SED['flux_duste'],lw=lw,color='darkred',label='dust emission', zorder=3)
			if add_agn == 1:
				plt.plot(spec_SED['wave']/1.0e+4,spec_SED['flux_agn'],lw=lw,color='darkgreen',label='AGN torus emission', zorder=4)

	elif decompose==0 or decompose==False:
		bfit_photo_fluxes = convert_unit_spec_from_ergscm2A(photo_cwave,bfit_photo['flux'],funit=funit)
		bfit_spec_wave = bfit_mod_spec['wave']
		bfit_spec_flux_tot = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['flux'],funit=funit)

	if wunit==0 or wunit=='angstrom':
		plt.plot(bfit_spec_wave,bfit_spec_flux_tot,lw=lw,color='black',label='total', zorder=5)
		plt.scatter(photo_cwave,bfit_photo_fluxes, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=6)
		plt.errorbar(photo_cwave,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o', zorder=7)
		plt.scatter(photo_cwave,obs_fluxes,s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=8)

	elif wunit==1 or wunit=='micron':
		plt.plot(bfit_spec_wave/1.0e+4,bfit_spec_flux_tot,lw=lw,color='black',label='total', zorder=5)
		plt.scatter(photo_cwave/1.0e+4,bfit_photo_fluxes, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=6)
		plt.errorbar(photo_cwave/1.0e+4,obs_fluxes,yerr=obs_flux_err,color=photo_color,markersize=1,fmt='o', zorder=7)
		plt.scatter(photo_cwave/1.0e+4,obs_fluxes,s=markersize, marker='s', lw=2, edgecolor=photo_color, color='none', zorder=8)

	f1.text(0.25, 0.9, "reduced $\chi^2 = %.3f$" % header_samplers['redcd_chi2'], verticalalignment='bottom', 
			horizontalalignment='right', transform=f1.transAxes, color='black', fontsize=20)

	if decompose==1 or decompose==True:
		if show_legend == True:
			plt.legend(fontsize=int(fontsize_legend), ncol=2, loc=loc_legend)

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

	if xticks is not None:
		plt.xticks(xticks)
	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	#if xrange is None:
	#	if wunit==0 or wunit=='angstrom':
	#		xmin = min(photo_cwave)*0.7
	#		xmax = max(photo_cwave)*1.3
	#		plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
	#	elif wunit==1 or wunit=='micron':
	#		plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
	#		xmin = min(photo_cwave)*0.7/1e+4
	#		xmax = max(photo_cwave)*1.3/1e+4
	#elif xrange is not None:
	#	plt.xlim(xrange[0],xrange[1])
	#	xmin = xrange[0]
	#	xmax = xrange[1]

	plt.xlim(xmin,xmax)

	# get residual:
	residuals = (obs_fluxes - bfit_photo_fluxes)/obs_fluxes

	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,residuals, s=80, marker='s', lw=3.0, color='gray')
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,residuals, s=80, marker='s', lw=3.0, color='gray')

	x = np.linspace(xmin,xmax,100)
	y = x-x
	plt.plot(x,y,lw=2,color='black',linestyle='--')
		
	plt.savefig(name_plot, bbox_inches='tight')

	return name_plot


def plot_SED_mcmc_photo(filters=None,obs_photo=None,bfit_photo=None,bfit_mod_spec=None,header_samplers=None,
	logscale_x=True,logscale_y=True,xrange=None,yrange=None,wunit='micron',funit='erg/s/cm2/A',decompose=1,
	xticks=None,photo_color='blue',residual_range=[-1.0,1.0],fontsize_tick=18,fontsize_label=28, 
	show_legend=True,loc_legend=2,fontsize_legend=20,markersize=100,lw=1.0,name_plot=None):

	from matplotlib.gridspec import GridSpec

	# observed SEDs
	nbands = len(filters)
	obs_fluxes = obs_photo['flux']
	obs_flux_err = obs_photo['flux_err']

	# central wavelength of all filters
	photo_cwave = cwave_filters(filters)

	# convert flux unit of observed SED
	obs_fluxes = convert_unit_spec_from_ergscm2A(photo_cwave,obs_fluxes,funit=funit)
	obs_flux_err = convert_unit_spec_from_ergscm2A(photo_cwave,obs_flux_err,funit=funit)

	fig1 = plt.figure(figsize=(14,9))
	gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], left=0.1, right=0.96, top=0.98, bottom=0.13, hspace=0.001)

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

	if yrange is None:
		plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
	if yrange is not None:
		plt.ylim(yrange[0],yrange[1])

	#if xrange is None:
	#	if wunit==0 or wunit=='angstrom':
	#		plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
	#		xmin = min(photo_cwave)*0.7
	#		xmax = max(photo_cwave)*1.3
	#	elif wunit==1 or wunit=='micron':
	#		plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
	#		xmin = min(photo_cwave)*0.7/1e+4
	#		xmax = max(photo_cwave)*1.3/1e+4

	#elif xrange is not None:
	#	plt.xlim(xrange[0],xrange[1])
	#	xmin = xrange[0]
	#	xmax = xrange[1]

	if xrange is None:
		xmin, xmax = min(photo_cwave)*0.7, max(photo_cwave)*1.3
	elif xrange is not None:
		xmin, xmax = xrange[0], xrange[1]

	if wunit==0 or wunit=='angstrom':
		plt.xlim(xmin,xmax)
	elif wunit==1 or wunit=='micron':
		xmin, xmax = xmin/1e+4, xmax/1e+4
		plt.xlim(xmin,xmax)

	#==> best-fit model spectrum
	if wunit==0 or wunit=='angstrom':
		spec_wave = bfit_mod_spec['wave']
	elif wunit==1 or wunit=='micron':
		spec_wave = bfit_mod_spec['wave']/1e+4

	p16_spec_tot = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['tot_p16'],funit=funit)
	p50_spec_tot = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['tot_p50'],funit=funit)
	p84_spec_tot = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['tot_p84'],funit=funit)

	f1.fill_between(spec_wave,p16_spec_tot,p84_spec_tot,facecolor='gray',alpha=0.5,zorder=9)
	if decompose==1 or decompose==True:
		plt.plot(spec_wave,p50_spec_tot,lw=lw,color='black',zorder=9,label='total')
	elif decompose==0 or decompose==False:
		plt.plot(spec_wave,p50_spec_tot,lw=lw,color='black',zorder=9)

	if decompose==1 or decompose==True:
		# stellar emission
		p16_spec_stellar = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['stellar_p16'],funit=funit)
		p50_spec_stellar = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['stellar_p50'],funit=funit)
		p84_spec_stellar = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['stellar_p84'],funit=funit)
		f1.fill_between(spec_wave,p16_spec_stellar,p84_spec_stellar,facecolor='orange',alpha=0.25,zorder=8)
		plt.plot(spec_wave,p50_spec_stellar,lw=lw,color='darkorange',zorder=8,label='stellar emission')

		# nebular emission
		if header_samplers['add_neb_emission'] == 1:
			p16_spec_nebe = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['nebe_p16'],funit=funit)
			p50_spec_nebe = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['nebe_p50'],funit=funit)
			p84_spec_nebe = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['nebe_p84'],funit=funit)
			f1.fill_between(spec_wave,p16_spec_nebe,p84_spec_nebe,facecolor='cyan',alpha=0.25,zorder=8)
			plt.plot(spec_wave,p50_spec_nebe,lw=lw,color='darkcyan',zorder=8,label='nebular emission')

		# dust emission
		if header_samplers['duste_stat']==1:
			p16_spec_duste = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['duste_p16'],funit=funit)
			p50_spec_duste = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['duste_p50'],funit=funit)
			p84_spec_duste = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['duste_p84'],funit=funit)
			f1.fill_between(spec_wave,p16_spec_duste,p84_spec_duste,facecolor='red',alpha=0.25,zorder=8)
			plt.plot(spec_wave,p50_spec_duste,lw=lw,color='darkred',zorder=8,label='dust emission')

		# AGN dusty torus emission
		if header_samplers['add_agn'] == 1:
			p16_spec_agn = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['agn_p16'],funit=funit)
			p50_spec_agn = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['agn_p50'],funit=funit)
			p84_spec_agn = convert_unit_spec_from_ergscm2A(bfit_mod_spec['wave'],bfit_mod_spec['agn_p84'],funit=funit)
			f1.fill_between(spec_wave,p16_spec_agn,p84_spec_agn,facecolor='green',alpha=0.25,zorder=8)
			plt.plot(spec_wave,p50_spec_agn,lw=lw,color='darkgreen',zorder=8,label='AGN torus emission')

		if show_legend == True:
			plt.legend(fontsize=int(fontsize_legend), loc=loc_legend, ncol=2)

	#==> plot best-fit photometric SED
	p50_photo_flux = convert_unit_spec_from_ergscm2A(photo_cwave,bfit_photo['p50'],funit=funit)

	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,p50_photo_flux, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9,alpha=0.5)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,p50_photo_flux, s=markersize, marker='s', lw=2, edgecolor='gray', color='none', zorder=9,alpha=0.5)

	#==> plot observed SED:
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

	if xticks is not None:
		plt.xticks(xticks)
	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	#if xrange is None:
	#	if wunit==0 or wunit=='angstrom':
	#		plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
	#		xmin = min(photo_cwave)*0.7
	#		xmax = max(photo_cwave)*1.3
	#	elif wunit==1 or wunit=='micron':
	#		plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
	#		xmin = min(photo_cwave)*0.7/1e+4
	#		xmax = max(photo_cwave)*1.3/1e+4
	#elif xrange is not None:
	#	plt.xlim(xrange[0],xrange[1])
	#	xmin = xrange[0]
	#	xmax = xrange[1]

	plt.xlim(xmin,xmax)

	# get residual:
	residuals = (obs_fluxes - p50_photo_flux)/obs_fluxes

	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,residuals, s=80, marker='s', lw=3.5, color='gray', zorder=9, alpha=1.0)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,residuals, s=80, marker='s', lw=3.5, color='gray', zorder=9, alpha=1.0)

	x = np.linspace(xmin,xmax,100)
	y = x-x
	plt.plot(x,y,lw=2,color='black',linestyle='--')
		
	plt.subplots_adjust(left=0.25, right=0.98, bottom=0.25, top=0.98)
	plt.savefig(name_plot, bbox_inches='tight')

	return name_plot

def plot_SED_specphoto_old(filters=None,obs_photo=None,obs_spec=None,bfit_photo=None,bfit_spec=None,bfit_mod_spec=None,minchi2_params=None,
	header_samplers=None,logscale_x=True, logscale_y=True, xrange=None, yrange=None, wunit='micron',funit='erg/s/cm2/A', 
	decompose=1,xticks=None,photo_color='red',residual_range=[-1.0,1.0], fontsize_tick=18,fontsize_label=25,show_legend=True, 
	loc_legend=4, fontsize_legend=18, markersize=100, lw=2.0, name_plot=None):

	# plot 1: photometry
	name_plot1 = 'ph_%s' % name_plot
	if header_samplers['fitmethod'] == 'mcmc':
		plot_SED_mcmc_photo(filters=filters,obs_photo=obs_photo,bfit_photo=bfit_photo,bfit_mod_spec=bfit_mod_spec,header_samplers=header_samplers,
				logscale_x=logscale_x,logscale_y=logscale_y,xrange=xrange,yrange=yrange,wunit=wunit,funit=funit,decompose=decompose,
				xticks=xticks,photo_color=photo_color,residual_range=residual_range,fontsize_tick=fontsize_tick,fontsize_label=fontsize_label, 
				show_legend=show_legend,loc_legend=loc_legend,fontsize_legend=fontsize_legend,markersize=markersize,lw=lw,name_plot=name_plot1)

	elif header_samplers['fitmethod'] == 'rdsps':
		plot_SED_rdsps_photo(filters=filters,obs_photo=obs_photo,bfit_photo=bfit_photo,bfit_mod_spec=bfit_mod_spec,minchi2_params=minchi2_params,
				header_samplers=header_samplers,logscale_x=logscale_x,logscale_y=logscale_y,xrange=xrange,yrange=yrange,wunit=wunit,funit=funit,
				decompose=decompose,xticks=xticks,photo_color=photo_color,residual_range=residual_range,fontsize_tick=fontsize_tick,
				fontsize_label=fontsize_label,show_legend=show_legend,loc_legend=loc_legend,fontsize_legend=fontsize_legend,
				markersize=markersize,lw=lw,name_plot=name_plot1)

	# plot 2: spectroscopy
	fig1 = plt.figure(figsize=(18,6))
	f1 = plt.subplot()
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))
	
	plt.ylim(min(obs_spec['flux'])*0.4,max(obs_spec['flux'])*1.8)
	xmin, xmax = min(obs_spec['wave'])-200, max(obs_spec['wave'])+200
	plt.xlim(xmin,xmax)

	plt.ylabel(r'$F_{\lambda}$ [erg $s^{-1}cm^{-2}\AA^{-1}$]', fontsize=int(fontsize_label))
	plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))

	if logscale_y == True:
		f1.set_yscale('log')
	if logscale_x == True:
		f1.set_xscale('log')

	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	f1.fill_between(obs_spec['wave'],obs_spec['flux']-obs_spec['flux_err'],obs_spec['flux']+obs_spec['flux_err'], 
					color='gray', alpha=0.2, edgecolor='none', zorder=3)
	plt.plot(obs_spec['wave'], obs_spec['flux'], lw=lw, color='black', label='Observed spectrum', zorder=4)

	if header_samplers['fitmethod'] == 'mcmc':
		f1.fill_between(bfit_spec['wave'], bfit_spec['p16'], bfit_spec['p84'], color='pink', alpha=0.2, edgecolor='none', zorder=5)
		plt.plot(bfit_spec['wave'], bfit_spec['p50'], lw=1, color='red', zorder=6)

	elif header_samplers['fitmethod'] == 'rdsps':
		plt.plot(bfit_spec['wave'], bfit_spec['flux'], lw=1, color='red', zorder=5)

	plt.subplots_adjust(bottom=0.2)
	name_plot2 = 'sp_%s' % name_plot
	plt.savefig(name_plot2, bbox_inches='tight')


	# plot 3: photometry + spectroscopy
	# observed SEDs
	nbands = len(filters)
	photo_cwave = cwave_filters(filters)
	obs_fluxes = convert_unit_spec_from_ergscm2A(photo_cwave,obs_photo['flux'],funit=funit)
	obs_flux_err = convert_unit_spec_from_ergscm2A(photo_cwave,obs_photo['flux_err'],funit=funit)
	obs_spec_flux = convert_unit_spec_from_ergscm2A(obs_spec['wave'],obs_spec['flux'],funit=funit)

	fig1 = plt.figure(figsize=(18,6))
	f1 = plt.subplot()
	plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
	plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))

	if logscale_y == True:
		f1.set_yscale('log')
	if logscale_x == True:
		f1.set_xscale('log')

	if funit=='erg/s/cm2/A' or funit==0:
		plt.ylabel(r'$F_{\lambda}$ [erg $s^{-1}cm^{-2}\AA^{-1}$]', fontsize=int(fontsize_label))
	elif funit=='erg/s/cm2' or funit==1:
		plt.ylabel(r'$\lambda F_{\lambda}$ [erg $s^{-1}cm^{-2}$]', fontsize=int(fontsize_label))
	elif funit=='Jy' or funit==2:
		plt.ylabel(r'$F_{\nu}$ [Jy]', fontsize=int(fontsize_label))
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if yrange is None:
		plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
	if yrange is not None:
		plt.ylim(yrange[0],yrange[1])

	if xrange is None:
		if wunit==0 or wunit=='angstrom':
			plt.xlim(min(photo_cwave)*0.7,max(photo_cwave)*1.3)
			xmin = min(photo_cwave)*0.7
			xmax = max(photo_cwave)*1.3
		elif wunit==1 or wunit=='micron':
			plt.xlim(min(photo_cwave)*0.7/1e+4,max(photo_cwave)*1.3/1e+4)
			xmin = min(photo_cwave)*0.7/1e+4
			xmax = max(photo_cwave)*1.3/1e+4
	elif xrange is not None:
		plt.xlim(xrange[0],xrange[1])
		xmin = xrange[0]
		xmax = xrange[1]

	if wunit==0 or wunit=='angstrom':
		plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
	elif wunit==1 or wunit=='micron':
		plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

	if xticks is not None:
		plt.xticks(xticks)
	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	#==> best-fit model spectrum
	if header_samplers['fitmethod'] == 'mcmc': 
		bfit_spec_flux = convert_unit_spec_from_ergscm2A(bfit_spec['wave'],bfit_spec['p50'],funit=funit)
		bfit_photo_flux = convert_unit_spec_from_ergscm2A(photo_cwave,bfit_photo['p50'],funit=funit)
	elif header_samplers['fitmethod'] == 'rdsps':
		bfit_spec_flux = convert_unit_spec_from_ergscm2A(bfit_spec['wave'],bfit_spec['flux'],funit=funit)
		bfit_photo_flux = convert_unit_spec_from_ergscm2A(photo_cwave,bfit_photo['flux'],funit=funit)

	if wunit==0 or wunit=='angstrom':
		plt.plot(obs_spec['wave'], obs_spec_flux, lw=lw, color='black', zorder=4)
		plt.plot(bfit_spec['wave'], bfit_spec_flux, lw=1, color='red', zorder=5)

		plt.scatter(photo_cwave, obs_fluxes, marker='o', s=markersize, color='blue', zorder=6)
		plt.scatter(photo_cwave, bfit_photo_flux, marker='o', s=0.7*markersize, color='gray', zorder=7)

	elif wunit==1 or wunit=='micron':
		plt.plot(obs_spec['wave']/1.0e+4, obs_spec_flux, lw=lw, color='black', zorder=4)
		plt.plot(bfit_spec['wave']/1.0e+4, bfit_spec_flux, lw=1, color='red', zorder=5)

		plt.scatter(photo_cwave/1.0e+4, obs_fluxes, marker='o', s=markersize, color='blue', zorder=6)
		plt.scatter(photo_cwave/1.0e+4, bfit_photo_flux, marker='o', s=0.7*markersize, color='gray', zorder=7)

	plt.subplots_adjust(bottom=0.2)
	name_plot3 = 'sph_%s' % name_plot
	plt.savefig(name_plot3, bbox_inches='tight')


def plot_SED_specphoto(filters=None,obs_photo=None,obs_spec=None,bfit_photo=None,bfit_spec=None,bfit_mod_spec=None,
	corr_factor=None,minchi2_params=None,header_samplers=None,logscale_x=True, logscale_y=True, xrange=None, yrange=None, 
	wunit='micron',funit='erg/s/cm2/A', xticks=None, photo_color='red',residual_range=[-1.0,1.0], show_original_spec=False,
	fontsize_tick=18, fontsize_label=25, show_legend=True, loc_legend=4, fontsize_legend=18, markersize=100, lw=2.0, name_plot=None):
	
	from matplotlib.gridspec import GridSpec
	from matplotlib.ticker import AutoMinorLocator

	gal_z = header_samplers['gal_z']
	del_wave_nebem = header_samplers['del_wave_nebem']

	photo_flux = obs_photo['flux']
	photo_flux_err = obs_photo['flux_err']

	photo_wave = bfit_photo['wave']
	bfit_photo_flux = bfit_photo['p50']

	obs_spec_wave = obs_spec['wave']
	obs_spec_flux = obs_spec['flux']
	obs_spec_flux_err = obs_spec['flux_err']

	bfit_spec_wave = bfit_spec['wave']
	bfit_spec_flux = bfit_spec['tot_p50']
	bfit_spec_flux_nebe = bfit_spec['nebe_p50']
	bfit_spec_flux_nebe_cont = bfit_spec['nebe_cont_p50']
	bfit_spec_flux_nebe_lines = bfit_spec_flux_nebe - bfit_spec_flux_nebe_cont

	bfit_mod_spec_wave = bfit_mod_spec['wave']
	bfit_mod_spec_flux = bfit_mod_spec['tot_p50']
	bfit_mod_spec_flux_nebe = bfit_mod_spec['nebe_p50']
	bfit_mod_spec_flux_nebe_cont = bfit_mod_spec['nebe_cont_p50']
	bfit_mod_spec_flux_nebe_lines = bfit_mod_spec_flux_nebe - bfit_mod_spec_flux_nebe_cont

	idx_sel = np.where((bfit_mod_spec_wave>=min(obs_spec_wave)) & (bfit_mod_spec_wave<=max(obs_spec_wave)))
	bfit_mod_spec_wave_cut = bfit_mod_spec_wave[idx_sel[0]]
	bfit_mod_spec_flux_cut = bfit_mod_spec_flux[idx_sel[0]]
	bfit_mod_spec_flux_nebe_cut = bfit_mod_spec_flux_nebe[idx_sel[0]]
	bfit_mod_spec_flux_nebe_cont_cut = bfit_mod_spec_flux_nebe_cont[idx_sel[0]]
	bfit_mod_spec_flux_nebe_lines_cut = bfit_mod_spec_flux_nebe_lines[idx_sel[0]]

	corr_factor_wave = corr_factor['wave']
	corr_factor_values = corr_factor['p50']

	func = interp1d(corr_factor_wave, corr_factor_values, fill_value='extrapolate')
	interp_corr_factor_values = func(bfit_mod_spec_wave_cut)

	rescaled_bfit_mod_spec_wave_cut = bfit_mod_spec_wave_cut
	rescaled_bfit_mod_spec_flux_cut = bfit_mod_spec_flux_cut*interp_corr_factor_values
	rescaled_bfit_mod_spec_flux_nebe_cut = bfit_mod_spec_flux_nebe_cut*interp_corr_factor_values
	rescaled_bfit_mod_spec_flux_nebe_cont_cut = bfit_mod_spec_flux_nebe_cont_cut*interp_corr_factor_values
	rescaled_bfit_mod_spec_flux_nebe_lines_cut = bfit_mod_spec_flux_nebe_lines_cut*interp_corr_factor_values

	rescaled_bfit_mod_spec_flux_no_lines_cut = rescaled_bfit_mod_spec_flux_cut-rescaled_bfit_mod_spec_flux_nebe_lines_cut

	# Calculate residuals
	obs_spec_wave_clean, waveid_excld = get_no_nebem_wave_fit(gal_z, obs_spec_wave, del_wave_nebem)
	waveid_incld = np.delete(np.arange(0,len(obs_spec_wave)), waveid_excld)

	photo_residuals = (photo_flux - bfit_photo_flux)/photo_flux
	spec_residuals = (obs_spec_flux[waveid_incld] - bfit_spec_flux[waveid_incld])/obs_spec_flux[waveid_incld]

	# flux unit:
	if funit != 'erg/s/cm2/A':
		if show_original_spec == True:
			bfit_mod_spec_flux1 = convert_unit_spec_from_ergscm2A(bfit_mod_spec_wave,bfit_mod_spec_flux,funit=funit)
		obs_spec_flux1 = convert_unit_spec_from_ergscm2A(obs_spec_wave,obs_spec_flux,funit=funit)
		rescaled_bfit_mod_spec_flux_no_lines_cut1 = convert_unit_spec_from_ergscm2A(rescaled_bfit_mod_spec_wave_cut,rescaled_bfit_mod_spec_flux_no_lines_cut,funit=funit)
		bfit_photo_flux1 = convert_unit_spec_from_ergscm2A(photo_wave,bfit_photo_flux,funit=funit)
		photo_flux1 = convert_unit_spec_from_ergscm2A(photo_wave,photo_flux,funit=funit)
		photo_flux_err1 = convert_unit_spec_from_ergscm2A(photo_wave,photo_flux_err,funit=funit)
	else:
		if show_original_spec == True:
			bfit_mod_spec_flux1 = bfit_mod_spec_flux
		obs_spec_flux1 = obs_spec_flux
		rescaled_bfit_mod_spec_flux_no_lines_cut1 = rescaled_bfit_mod_spec_flux_no_lines_cut
		bfit_photo_flux1 = bfit_photo_flux
		photo_flux1 = photo_flux
		photo_flux_err1 = photo_flux_err

	# wavelength units:
	if wunit==1 or wunit=='micron':
		bfit_mod_spec_wave1 = bfit_mod_spec_wave/1e+4
		obs_spec_wave1 = obs_spec_wave/1e+4
		rescaled_bfit_mod_spec_wave_cut1 = rescaled_bfit_mod_spec_wave_cut/1e+4
		photo_wave1 = photo_wave/1e+4
	else:
		bfit_mod_spec_wave1 = bfit_mod_spec_wave
		obs_spec_wave1 = obs_spec_wave
		rescaled_bfit_mod_spec_wave_cut1 = rescaled_bfit_mod_spec_wave_cut
		photo_wave1 = photo_wave


	###==> plotting
	fig1 = plt.figure(figsize=(15,9))
	gs = GridSpec(nrows=2, ncols=1, height_ratios=[3,1], left=0.1, right=0.96, top=0.92, bottom=0.13, hspace=0.001)

	f1 = fig1.add_subplot(gs[0])
	plt.setp(f1.get_xticklabels(), visible=False)
	plt.setp(f1.get_yticklabels(), fontsize=15)

	if xrange is None:
		bulk_waves = photo_wave1.tolist() + obs_spec_wave1.tolist()
		xmin, xmax = 0.7*min(bulk_waves), 1.1*max(bulk_waves)
	else:
		xmin, xmax = xrange[0], xrange[1]

	if yrange is None:
		bulk_fluxes = photo_flux1.tolist() + bfit_photo_flux1.tolist() + obs_spec_flux1.tolist() + rescaled_bfit_mod_spec_flux_no_lines_cut1.tolist()
		ymin, ymax = 0.7*min(bulk_fluxes), 1.2*max(bulk_fluxes)
	else:
		ymin, ymax = yrange[0], yrange[1]
	plt.ylim(ymin,ymax)

	if logscale_y == True:
		f1.set_yscale('log')
	if logscale_x == True:
		f1.set_xscale('log')

	#if xticks is not None:
	#	plt.xticks(xticks)
	#for axis in [f1.xaxis]:
	#	axis.set_major_formatter(ScalarFormatter())

	plt.xlim(xmin, xmax)

	if funit=='erg/s/cm2/A' or funit==0:
		plt.ylabel(r'$F_{\lambda}$ [erg $\rm{s}^{-1}\rm{cm}^{-2}\AA^{-1}$]', fontsize=int(fontsize_label))
	elif funit=='erg/s/cm2' or funit==1:
		plt.ylabel(r'$\lambda F_{\lambda}$ [erg $\rm{s}^{-1}\rm{cm}^{-2}$]', fontsize=int(fontsize_label))
	elif funit=='Jy' or funit==2:
		plt.ylabel(r'$F_{\nu}$ [Jy]', fontsize=int(fontsize_label))
	else:
		print ("The input funit is not recognized!")
		sys.exit()

	if show_original_spec == True:
		plt.plot(bfit_mod_spec_wave1, bfit_mod_spec_flux1, lw=0.5, color='lightblue', zorder=0)
	plt.plot(obs_spec_wave1, obs_spec_flux1, lw=1, color='black', zorder=1, label='Observed spectrum')
	plt.plot(rescaled_bfit_mod_spec_wave_cut1, rescaled_bfit_mod_spec_flux_no_lines_cut1, lw=1, color='red', zorder=2, label='Model continuum')

	plt.scatter(photo_wave1, bfit_photo_flux1, s=100, marker='s', edgecolor='blue', color='none', lw=3, zorder=3, label='Model photometry')
	plt.errorbar(photo_wave1, photo_flux1, yerr=photo_flux_err1, fmt='o', color='green', markersize=10, lw=2, zorder=4, label='Observed photometry')

	plt.legend(fontsize=17)

	f2 = f1.twiny()
	if wunit==1 or wunit=='micron':
		f2.set_xlabel(r'Rest-frame wavelength [$\mu$m]', fontsize=18)
	else:
		f2.set_xlabel(r'Rest-frame wavelength [$\AA$]', fontsize=18)
	plt.setp(f2.get_xticklabels(), fontsize=14)
	plt.xlim(xmin/(1.0+gal_z), xmax/(1.0+gal_z))
	if logscale_x == True:
		f2.set_xscale('log')
	f2.xaxis.set_minor_locator(AutoMinorLocator())


	### plot residual
	f1 = fig1.add_subplot(gs[1])
	plt.setp(f1.get_yticklabels(), fontsize=15)
	plt.setp(f1.get_xticklabels(), fontsize=15)
	plt.ylabel(r'residual', fontsize=22)
	if wunit==1 or wunit=='micron':
		f1.set_xlabel(r'Observed wavelength [$\mu$m]', fontsize=22)
	else:
		f1.set_xlabel(r'Observed wavelength [$\AA$]', fontsize=22)
	plt.ylim(residual_range[0],residual_range[1])

	if logscale_x == True:
		f1.set_xscale('log')

	if xticks is not None:
		plt.xticks(xticks)
	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())
	plt.xlim(xmin,xmax)

	f1.xaxis.set_minor_locator(AutoMinorLocator())

	plt.plot(obs_spec_wave1[waveid_incld], spec_residuals, lw=1, color='gray', zorder=0)
	plt.scatter(photo_wave1, photo_residuals, s=100, marker='s', edgecolor='black', color='none', lw=4, zorder=1)

	x = np.linspace(xmin,xmax,200)
	plt.plot(x,x-x,lw=1,color='black',linestyle='--')

	plt.savefig(name_plot, bbox_inches='tight')
	return name_plot


def plot_SED(name_sampler_fits,logscale_x=False,logscale_y=True,xrange=None,yrange=None,wunit='micron',funit='erg/s/cm2/A', 
	decompose=0,xticks=None,photo_color='red',residual_range=[-1.0,1.0],show_original_spec=False,fontsize_tick=18,fontsize_label=25,
	show_legend=True, loc_legend=4, fontsize_legend=18, markersize=100, lw=2.0, name_plot=None):
	"""Function for plotting best-fit (i.e., median posterior) model SED from a result of SED fitting. 

	:param name_sampler_fits:
		Name of input FITS file containing result of an SED fitting. 

	:param logscale_x: 
		Choice for plotting x-axis in logarithmic (True) or linear scale (False).

	:param logscale_y: 
		Choice for plotting y-axis in logarithmic (True) or linear scale (False).

	:param xrange: 
		Range in x-axis. The accepted format is: [xmin,xmax]. If xrange=None, the range will be defined based on 
		the wavelength range of the observed SED.

	:param yrange: 
		Range in y-axis. The accepted format is: [ymin,ymax]. If yrange=None, the range will be defined based on
		the fluxes range of the observed SED.  
	
	:param wunit: 
		Wavelength unit. Options are: 0 or 'angstrom' for Angstrom unit and 1 or 'micron' for micron unit.

	:param funit: 
		Flux unit. Options are: 0 or 'erg/s/cm2/A', 1 or 'erg/s/cm2', and 2 or 'Jy'.

	:param decompose: 
		Choice for showing best-fit (i.e., median posterior) model SED broken down into its components (1 or True) or just its total (0 or False).
	
	:param xticks: 
		xticks in list format.

	:param photo_color: 
		Color of photometric fluxes data points. The accepted colors are those available in the matplotlib.

	:param residual_range: 
		Residuals between observed SED and the median posterior model SED. 
		The residual is defined as (D - M)/D, where D represents observed SED, while M is model SED.

	:param show_original_spec: (default=False)
		Show original best-fit model spectrum before rescaling with polnomial correction. This is only relevant if the data is spectrophotometric.

	:param fontsize_tick: 
		Fontsize for the ticks. 
	
	:param fontsize_label: 
		Fontsize for the labels in the x and y axes. 

	:param show_legend: 
		Option for showing a legend.

	:param loc_legend: 
		Location of the legend.

	:param fontsize_legend: 
		Fontsize of the legend.

	:param markersize: 
		Size for the markers of the observed and model SEDs.

	:param lw: (optional, default: 1)
		Line width for the best-fit (i.e., median posterior) model SED.
	
	:param name_plot: 
		Desired name for the output plot. This is optional. If None, a default name will be used. 
	"""

	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	obs_photo = hdu['obs_photo'].data
	bfit_photo = hdu['bfit_photo'].data
	if header_samplers['fitmethod'] == 'rdsps':
		minchi2_params = hdu['minchi2_params'].data
	if header_samplers['specphot'] == 0:
		bfit_mod_spec = hdu['bfit_mod_spec'].data 
	if header_samplers['specphot'] == 1:
		obs_spec = hdu['obs_spec'].data
		bfit_spec = hdu['bfit_spec'].data
		bfit_mod_spec = hdu['bfit_mod'].data
		corr_factor = hdu['corr_factor'].data 
	hdu.close()

	# filters
	nbands = int(header_samplers['nfilters'])
	filters = []
	for bb in range(0,nbands):
		filters.append(header_samplers['fil%d' % bb])

	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "sed_%s.png" % (name_sampler_fits1)

	if header_samplers['specphot'] == 1:
		if header_samplers['fitmethod'] == 'mcmc':
			minchi2_params = None
		plot_SED_specphoto(filters=filters,obs_photo=obs_photo,obs_spec=obs_spec,bfit_photo=bfit_photo,bfit_spec=bfit_spec,
			bfit_mod_spec=bfit_mod_spec,corr_factor=corr_factor,minchi2_params=minchi2_params,header_samplers=header_samplers,
			logscale_x=logscale_x,logscale_y=logscale_y,xrange=xrange,yrange=yrange,wunit=wunit,funit=funit,xticks=xticks,
			photo_color=photo_color,residual_range=residual_range,show_original_spec=show_original_spec,fontsize_tick=fontsize_tick,
			fontsize_label=fontsize_label,show_legend=show_legend,loc_legend=loc_legend,fontsize_legend=fontsize_legend,
			markersize=markersize,lw=lw,name_plot=name_plot)
	elif header_samplers['specphot'] == 0:
		if header_samplers['fitmethod'] == 'mcmc':
			plot_SED_mcmc_photo(filters=filters,obs_photo=obs_photo,bfit_photo=bfit_photo,bfit_mod_spec=bfit_mod_spec,
				header_samplers=header_samplers,logscale_x=logscale_x,logscale_y=logscale_y,xrange=xrange,yrange=yrange,
				wunit=wunit,funit=funit,decompose=decompose,xticks=xticks,photo_color=photo_color,residual_range=residual_range,
				fontsize_tick=fontsize_tick,fontsize_label=fontsize_label,show_legend=show_legend,loc_legend=loc_legend,
				fontsize_legend=fontsize_legend,markersize=markersize,lw=lw,name_plot=name_plot)

		elif header_samplers['fitmethod'] == 'rdsps':
			plot_SED_rdsps_photo(filters=filters,obs_photo=obs_photo,bfit_photo=bfit_photo,bfit_mod_spec=bfit_mod_spec,minchi2_params=minchi2_params,
				header_samplers=header_samplers,logscale_x=logscale_x,logscale_y=logscale_y,xrange=xrange,yrange=yrange,wunit=wunit,
				funit=funit,decompose=decompose,xticks=xticks,photo_color=photo_color,residual_range=residual_range,fontsize_tick=fontsize_tick,
				fontsize_label=fontsize_label,show_legend=show_legend,loc_legend=loc_legend,fontsize_legend=fontsize_legend,markersize=markersize,
				lw=lw,name_plot=name_plot)



def plot_corner(name_sampler_fits, params=['log_sfr','log_mass','log_dustmass','log_fagn','log_fagn_bol','log_tauagn',
	'log_qpah','log_umin','log_gamma','dust1','dust2','dust_index','log_mw_age','log_age','log_t0','log_alpha','log_beta',
	'log_tau','logzsol','z','gas_logu','gas_logz'], label_params={'log_sfr':'log(SFR)','log_mass':'log($M_{*}$)','log_dustmass':'log($M_{dust}$)',
	'log_fagn':'log($f_{AGN,*}$)','log_fagn_bol':'log($f_{AGN,bol}$)','log_tauagn':'log($\\tau_{AGN}$)','log_qpah':'log($Q_{PAH}$)',
	'log_umin':'log($U_{min}$)','log_gamma':'log($\gamma_{e}$)','dust1':'$\hat \\tau_{1}$','dust2':'$\hat \\tau_{2}$', 'dust_index':'$n$', 
	'log_mw_age':'log($\mathrm{age}_{\mathrm{M}}$)','log_age':'log($\mathrm{age}_{\mathrm{sys}}$)','log_t0':'log($t_{0}$)',
	'log_alpha':'log($\\alpha$)', 'log_beta':'log($\\beta$)','log_tau':'log($\\tau$)','logzsol':'log($Z/Z_{\odot}$)','z':'z', 
	'gas_logu':'log($U$)', 'gas_logz':'log($Z_{gas}/Z_{\odot}$)'}, 
	params_ranges = {'log_sfr':[-99.0,-99.0],'log_mass':[-99.0,-99.0],'log_dustmass':[-99.0,-99.0],'log_fagn':[-5.0,0.48],
	'log_fagn_bol':[-99.0,-99.0],'log_tauagn':[0.70,2.18],'log_qpah':[-1.0, 0.845],'log_umin':[-1.0, 1.176],'log_gamma':[-3.0,-0.824],
	'dust1':[0.0,4.0],'dust2':[0.0,4.0], 'dust_index':[-2.2,0.4],'log_mw_age':[-99.0,-99.0],'log_age': [-3.0, 1.14],
	'log_t0': [-2.0, 1.14],'log_alpha':[-2.5,2.5],'log_beta':[-2.5,2.5],'log_tau': [-2.5, 1.5], 'logzsol': [-2.0, 0.5], 
	'z': [-99.0, -99.0],'gas_logu':[-4.0,-1.0],'gas_logz':[-2.0,0.2]}, factor=1.0, nbins=12, fontsize_label=20, fontsize_tick=14, name_plot=None):
	
	"""Function for producing corner plot that shows 1D and joint 2D posterior probability distributions from a fitting result with the MCMC method.
	
	:param name_sampler_fits:
		Name of input FITS file containing result of an SED fitting.

	:param params: 
		List of parameters to be shown in the corner plot. This is optional. 
		If default input is used, all the parameters incolved in the SED fitting will be shown in the corner plot.

	:param label_params: 
		Labels for the parameters. The accepted format is a python dictionary.

	:param params_ranges: 
		Desired ranges for the parameters.

	:param factor:
		Multiplication factor to be applied to stellar mass, SFR, and dust mass.

	:param nbins: 
		Number of binning in the parameter space when calculating the joint posteriors.

	:param fontsize_label:
		Fontsize of labels in the x and y axes.

	:param fontsize_tick:
		Fontsize for the ticks.

	:param name_plot: (optional, default: None)
		Desired name for the output plot. 

	:returns name_plot:
		Desired name for the output plot. This is optional. If None, a default name will be used.
	"""

	def_params0 = list_default_params_fit()
	def_params = def_params0 + ['log_sfr', 'log_mass', 'log_mw_age', 'log_dustmass']

	# open the input FITS file
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[1].header
	data_samplers = hdu[1].data
	if hdu[0].header['storesamp'] == 0:
		print ("The input FITS file does not contain sampler chains!")
		sys.exit()
	hdu.close()

	# number of parameters
	nparams = len(params)
	if nparams == len(def_params):                          # if default set is used 
		nparams_fit = int(header_samplers['TFIELDS']) - 1
		params_new = []
		label_params_new = {}
		params_ranges_new = {}
		for ii in range(0,nparams):
			for jj in range(0,nparams_fit):
				str_temp = 'TTYPE%d' % (jj+2)
				if params[ii] == header_samplers['TTYPE%d' % (jj+2)]:
					params_new.append(params[ii])
					label_params_new[params[ii]] = label_params[params[ii]]
					params_ranges_new[params[ii]] = params_ranges[params[ii]]
	else:
		params_new = params
		label_params_new = label_params
		params_ranges_new = params_ranges

	# apply multiplication factor 
	data_samplers['log_mass'] = data_samplers['log_mass'] + np.log10(factor) 
	data_samplers['log_sfr'] = data_samplers['log_sfr'] + np.log10(factor)
	if 'log_dustmass' in params_new:
		data_samplers['log_dustmass'] = data_samplers['log_dustmass'] + np.log10(factor)

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


def plot_sfh_mcmc(name_sampler_fits, nchains=200, del_t=0.05, lbacktime_max=None, yrange=None, factor=1.0, loc_legend=2, fontsize_tick=18, 
	fontsize_label=25, fontsize_legend=26, logscale_x=False, logscale_y=False, name_plot=None):
	"""Function for producing SFH plot from a fitting result with the MCMC method. This is only applicable for fitting result 
	that stores the full sampler chains, which is when we set store_full_samplers=1 in the SED fitting functions. 

	:param name_sampler_fits:
		Name of input FITS file containing result of an SED fitting.

	:param nchains:
		Number of randomly chosen sampler chains (from the full samplers stored in the FITS file) to be used for calculating the inferred SFH.

	:param del_t:
		Width of the look back time binning in unit of Gyr for sampling the star formation history (SFH).

	:param lbacktime_max: 
		Maximum look-back time in the SFH plot. If None, the maximum look-back time is defined from the age of universe at the redshift of the galaxy.

	:param yrange: 
		Range in the y-axis.

	:param factor:
		Multiplication factor to be applied to the SFH.

	:param loc_legend: 
		Where to locate the legend. This is the same as in the `matplotlib`.

	:param fontsize_tick: 
		Fontsize for the ticks.
	
	:param fontsize_label: 
		Fontsize of the labels in the x and y axes. 

	:param fontsize_legend: 
		Fontsize of the legend.

	:param logscale_x: 
		Choice for plotting x-axis in logarithmic (True) or linear scale (False).

	:param logscale_y: 
		Choice for plotting y-axis in logarithmic (True) or linear scale (False).

	:returns name_plot:
		Desired name for the output plot. This is optional. If None, a default name will be used.

	:returns grid_lbt:
		Look-back times.

	:return grid_sfr_p16:
		16th percentile of the SFR(t).

	:return grid_sfr_p50:
		50th percentile of the SFR(t).

	:return grid_sfr_p84:
		84th percentile of the SFR(t).
	"""
	
	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	data_samplers = hdu[1].data
	hdu.close()

	sfh_form = header_samplers['sfh_form']
	nsamplers = len(data_samplers['log_age'])
	free_z = int(header_samplers['free_z'])

	# cosmology parameter
	cosmo = header_samplers['cosmo']
	H0 = float(header_samplers['H0'])
	Om0 = float(header_samplers['Om0'])

	if free_z == 0:
		gal_z = float(header_samplers['gal_z'])
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
	elif free_z == 1:
		max_z = max(data_samplers['z'])
		if cosmo == 'flat_LCDM':
			cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
			max_lbt = cosmo1.age(max_z).value
		elif cosmo == 'WMAP5':
			max_lbt = WMAP5.age(max_z).value
		elif cosmo == 'WMAP7':
			max_lbt = WMAP7.age(max_z).value
		elif cosmo == 'WMAP9':
			max_lbt = WMAP9.age(max_z).value
		elif cosmo == 'Planck13':
			max_lbt = Planck13.age(max_z).value
		elif cosmo == 'Planck15':
			max_lbt = Planck15.age(max_z).value
		#elif cosmo == 'Planck18':
		#	max_lbt = Planck18.age(gal_z).value

	nt = int(max_lbt/del_t)
	grid_lbt = np.linspace(0.0,max_lbt,nt)
	array_sfr_at_lbt = np.zeros((nchains,nt))

	## exclude saturated samplers: log(SFR)~-29.99..
	idx_sel = np.where(data_samplers['log_sfr']>-29.0)
	rand_idx = np.random.uniform(0,len(idx_sel[0]),nchains)

	for ii in range(0,nchains):
		idx = idx_sel[0][int(rand_idx[ii])]

		if free_z == 1:
			gal_z = data_samplers['z'][idx]
			if cosmo == 'flat_LCDM':
				cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
				max_lbt1 = cosmo1.age(gal_z).value
			elif cosmo == 'WMAP5':
				max_lbt1 = WMAP5.age(gal_z).value
			elif cosmo == 'WMAP7':
				max_lbt1 = WMAP7.age(gal_z).value
			elif cosmo == 'WMAP9':
				max_lbt1 = WMAP9.age(gal_z).value
			elif cosmo == 'Planck13':
				max_lbt1 = Planck13.age(gal_z).value
			elif cosmo == 'Planck15':
				max_lbt1 = Planck15.age(gal_z).value
		elif free_z == 0:
			max_lbt1 = max_lbt

		age = pow(10.0,data_samplers['log_age'][idx])
		tau = pow(10.0,data_samplers['log_tau'][idx])
		t0 = 0.0
		alpha = 0.0
		beta = 0.0
		if sfh_form==2 or sfh_form==3:
			t0 = pow(10.0,data_samplers['log_t0'][idx])
		if sfh_form==4:
			alpha = pow(10.0,data_samplers['log_alpha'][idx])
			beta = pow(10.0,data_samplers['log_beta'][idx])

		formed_mass = pow(10.0,data_samplers['log_mass'][idx])

		t,SFR_t = construct_SFH(sfh_form=sfh_form,t0=t0,tau=tau,alpha=alpha,beta=beta,age=age,formed_mass=formed_mass)
		t_back = np.abs(t - age)

		array_sfr_at_lbt[ii] = np.interp(grid_lbt,t_back[::-1],SFR_t[::-1],left=0,right=0)*factor

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

	plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "sfh_%s.png" % (name_sampler_fits1)
	
	plt.savefig(name_plot, bbox_inches='tight')

	return name_plot, grid_lbt, grid_sfr_p16, grid_sfr_p50, grid_sfr_p84

