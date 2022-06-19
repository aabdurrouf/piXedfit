import numpy as np
from math import pow
import sys, os
import fsps
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

from ..piXedfit_model import generate_modelSED_spec_decompose, construct_SFH, convert_unit_spec_from_ergscm2A
from ..utils.filtering import cwave_filters, filtering
from ..utils.posteriors import plot_triangle_posteriors


__all__ = ["plot_SED", "plot_corner", "plot_sfh_mcmc"]


def plot_SED_rdsps_photo(filters=None,obs_photo=None,bfit_photo=None,bfit_mod_spec=None,minchi2_params=None,header_samplers=None,
	logscale_x=True,logscale_y=True,xrange=None,yrange=None,wunit='micron',funit='erg/s/cm2/A',decompose=1,xticks=None,
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

	if yrange == None:
		plt.ylim(min(obs_fluxes)*0.5,max(obs_fluxes)*1.8)
	if yrange != None:
		plt.ylim(yrange[0],yrange[1])

	if decompose==1 or decompose==True:
		def_params = ['logzsol','log_tau','log_t0','log_alpha','log_beta', 'log_age','dust_index','dust1','dust2',
						'log_gamma','log_umin', 'log_qpah', 'z', 'log_fagn','log_tauagn', 'log_mass']

		def_params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,
						'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,
						'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}

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
										H0=H0,Om0=Om0,sfh_form=sfh_form,funit=funit)

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
	residuals = (obs_fluxes - bfit_photo_fluxes)/obs_fluxes

	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,residuals, s=80, marker='s', lw=3.0, color='gray')
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,residuals, s=80, marker='s', lw=3.0, color='gray')

	x = np.linspace(xmin,xmax,100)
	y = x-x
	plt.plot(x,y,lw=2,color='black',linestyle='--')
		
	plt.savefig(name_plot)

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
	residuals = (obs_fluxes - p50_photo_flux)/obs_fluxes

	if wunit==0 or wunit=='angstrom':
		plt.scatter(photo_cwave,residuals, s=80, marker='s', lw=3.5, color='gray', zorder=9, alpha=1.0)
	elif wunit==1 or wunit=='micron':
		plt.scatter(photo_cwave/1.0e+4,residuals, s=80, marker='s', lw=3.5, color='gray', zorder=9, alpha=1.0)

	x = np.linspace(xmin,xmax,100)
	y = x-x
	plt.plot(x,y,lw=2,color='black',linestyle='--')
		
	plt.subplots_adjust(left=0.25, right=0.98, bottom=0.25, top=0.98)
	plt.savefig(name_plot)

	return name_plot

def plot_SED_specphoto(filters=None,obs_photo=None,obs_spec=None,bfit_photo=None,bfit_spec=None,bfit_mod_spec=None,minchi2_params=None,
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
		f1.fill_between(bfit_spec['wave'], bfit_spec['p16'], bfit_spec['p84'], color='pink', alpha=0.5, zorder=5)
		plt.plot(bfit_spec['wave'], bfit_spec['p50'], lw=lw, color='red', zorder=6)

	elif header_samplers['fitmethod'] == 'rdsps':
		plt.plot(bfit_spec['wave'], bfit_spec['flux'], lw=lw, color='red', zorder=5)

	name_plot2 = 'sp_%s' % name_plot
	plt.savefig(name_plot2)


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

	if wunit==0 or wunit=='angstrom':
		plt.xlabel(r'Wavelength $[\AA]$', fontsize=int(fontsize_label))
	elif wunit==1 or wunit=='micron':
		plt.xlabel(r'Wavelength [$\mu$m]', fontsize=int(fontsize_label))

	if xticks != None:
		plt.xticks(xticks)
	for axis in [f1.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	#==> best-fit model spectrum
	if header_samplers['fitmethod'] == 'mcmc': 
		bfit_spec_flux = convert_unit_spec_from_ergscm2A(bfit_spec['wave'],bfit_spec['p50'],funit=funit)
		bfit_photo_flux = convert_unit_spec_from_ergscm2A(bfit_photo['wave'],bfit_photo['p50'],funit=funit)
	elif header_samplers['fitmethod'] == 'rdsps':
		bfit_spec_flux = convert_unit_spec_from_ergscm2A(bfit_spec['wave'],bfit_spec['flux'],funit=funit)
		bfit_photo_flux = convert_unit_spec_from_ergscm2A(bfit_photo['wave'],bfit_photo['flux'],funit=funit)

	if wunit==0 or wunit=='angstrom':
		plt.plot(obs_spec['wave'], obs_spec_flux, lw=lw, color='black', zorder=4)
		plt.plot(bfit_spec['wave'], bfit_spec_flux, lw=lw, color='red', zorder=5)

		plt.scatter(photo_cwave, obs_fluxes, marker='o', s=markersize, color='blue', zorder=6)
		plt.scatter(photo_cwave, bfit_photo_flux, marker='o', s=0.7*markersize, color='gray', zorder=7)

	elif wunit==1 or wunit=='micron':
		plt.plot(obs_spec['wave']/1.0e+4, obs_spec_flux, lw=lw, color='black', zorder=4)
		plt.plot(bfit_spec['wave']/1.0e+4, bfit_spec_flux, lw=lw, color='red', zorder=5)

		plt.scatter(photo_cwave/1.0e+4, obs_fluxes, marker='o', s=markersize, color='blue', zorder=6)
		plt.scatter(photo_cwave/1.0e+4, bfit_photo_flux, marker='o', s=0.7*markersize, color='gray', zorder=7)

	name_plot3 = 'sph_%s' % name_plot
	plt.savefig(name_plot3)



def plot_SED(name_sampler_fits,logscale_x=True,logscale_y=True,xrange=None,yrange=None,wunit='micron',funit='erg/s/cm2/A', 
	decompose=1,xticks=None,photo_color='red',residual_range=[-1.0,1.0],fontsize_tick=18,fontsize_label=25,show_legend=True, 
	loc_legend=4, fontsize_legend=18, markersize=100, lw=2.0, name_plot=None):
	"""Function for plotting best-fit model SED from a fitting result. 

	:param name_sampler_fits:
		Name of input FITS file containing sampler chains from the MCMC fitting. This FITS file must be output of :func:`singleSEDfit` or :func:`SEDfit_from_binmap` functions.

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

	:param photo_color: (optional, default: 'blue')
		Color of photometric fluxes points (in string). The accepted input is the same as that available in matplotlib.

	:param residual_range: (default: [-1.0,1.0])
		Residuals between observed SED and the median posterior model SED. 
		The residual in each band is defined as (f_D - f_M)/f_D, where f_D is flux in observed SED and f_M is flux in model SED.

	:param fontsize_tick: (optional, default: 20)
		Fontsize for the tick. Only relevant if xticks is not None. 
	
	:param fontsize_label: (optional, default: 25)
		Fontsize for the x- and y-axis labels. 

	:param show_legend: (optional, default: True)
		Flag whether to show legend or not.

	:param loc_legend: (optional, default: 2)
		Location of the legend.

	:param fontsize_legend: (optional, default: 18)
		Fontsize for the legend.

	:param markersize: (optional, default: 100)
		Size of the maarkers associated with the observed and model SEDs.

	:param lw: (optional, default: 1)
		Line width of the model SEDs.
	
	:param name_plot: (optional, default: None)
		Name of the output plot. This is optional parameter.
	"""

	hdu = fits.open(name_sampler_fits)
	header_samplers = hdu[0].header
	obs_photo = hdu['obs_photo'].data
	bfit_photo = hdu['bfit_photo'].data
	bfit_mod_spec = hdu['bfit_mod_spec'].data
	if header_samplers['specphot'] == 1:
		obs_spec = hdu['obs_spec'].data
		bfit_spec = hdu['bfit_spec'].data
	if header_samplers['fitmethod'] == 'rdsps':
		minchi2_params = hdu['minchi2_params'].data
	hdu.close()

	# filters
	nbands = int(header_samplers['nfilters'])
	filters = []
	for bb in range(0,nbands):
		filters.append(header_samplers['fil%d' % bb])

	if name_plot==None:
		name_sampler_fits1 = name_sampler_fits.replace('.fits','')
		name_plot = "sed_photo_%s.png" % (name_sampler_fits1)

	if header_samplers['specphot'] == 1:
		if header_samplers['fitmethod'] == 'mcmc':
			minchi2_params = None
		plot_SED_specphoto(filters=filters,obs_photo=obs_photo,obs_spec=obs_spec,bfit_photo=bfit_photo,bfit_spec=bfit_spec,
			bfit_mod_spec=bfit_mod_spec,minchi2_params=minchi2_params,header_samplers=header_samplers,logscale_x=logscale_x,
			logscale_y=logscale_y,xrange=xrange,yrange=yrange,wunit=wunit,funit=funit,decompose=decompose,xticks=xticks,
			photo_color=photo_color,residual_range=residual_range,fontsize_tick=fontsize_tick,fontsize_label=fontsize_label,
			show_legend=show_legend,loc_legend=loc_legend,fontsize_legend=fontsize_legend,markersize=markersize,lw=lw,
			name_plot=name_plot)
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
	'log_tau','logzsol','z'], label_params={'log_sfr':'log(SFR)','log_mass':'log($M_{*}$)',
	'log_dustmass':'log($M_{dust}$)','log_fagn':'log($f_{AGN,*}$)','log_fagn_bol':'log($f_{AGN,bol}$)',
	'log_tauagn':'log($\\tau_{AGN}$)','log_qpah':'log($Q_{PAH}$)','log_umin':'log($U_{min}$)','log_gamma':'log($\gamma_{e}$)',
	'dust1':'$\hat \\tau_{1}$','dust2':'$\hat \\tau_{2}$', 'dust_index':'$n$', 'log_mw_age':'log($\mathrm{age}_{\mathrm{MW}}$)',
	'log_age':'log($\mathrm{age}_{\mathrm{sys}}$)','log_t0':'log($t_{0}$)','log_alpha':'log($\\alpha$)', 
	'log_beta':'log($\\beta$)','log_tau':'log($\\tau$)','logzsol':'log($Z/Z_{\odot}$)','z':'z'}, 
	params_ranges = {'log_sfr':[-99.0,-99.0],'log_mass':[-99.0,-99.0],'log_dustmass':[-99.0,-99.0],'log_fagn':[-5.0,0.48],
	'log_fagn_bol':[-99.0,-99.0],'log_tauagn':[0.70,2.18],'log_qpah':[-1.0, 0.845],'log_umin':[-1.0, 1.176],'log_gamma':[-3.0,-0.824],
	'dust1':[0.0,3.0],'dust2':[0.0, 3.0], 'dust_index':[-2.2,0.4],'log_mw_age':[-99.0,-99.0],'log_age': [-3.0, 1.14],
	'log_t0': [-2.0, 1.14],'log_alpha':[-2.5,2.5],'log_beta':[-2.5,2.5],'log_tau': [-2.5, 1.5], 'logzsol': [-2.0, 0.5], 
	'z': [-99.0, -99.0]}, nbins=12, fontsize_label=20, fontsize_tick=14, name_plot=None):
	
	"""Function for producing corner plot that shows 1D and joint 2D posterior probability distributions from the fitting results with MCMC method.
	
	:param name_sampler_fits:
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

	if header_samplers['storesamp'] == 0:
		print ("The input FITS file does not contain sampler chains!")
		sys.exit()

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


def plot_sfh_mcmc(name_sampler_fits, nchains=200, del_t=0.05, plot_true=0, true_params = {'log_tau': -99.0, 'log_age': -99.0, 
	'log_t0': -99.0, 'log_alpha':-99.0, 'log_beta':-99.0, 'log_mass': -99.0}, true_SFH_lbt=[], true_SFH_sfr=[],lbacktime_max=None, 
	yrange=None, loc_legend=2, fontsize_tick=18, fontsize_label=25, fontsize_legend=26, logscale_x=False, logscale_y=False, name_plot=None):
	"""Function for producing SFH plot from fitting result obtained with the MCMC method.

	:param name_sampler_fits:
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

		array_sfr_at_lbt[ii] = np.interp(grid_lbt,t_back[::-1],SFR_t[::-1],left=0,right=0)

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
			true_tau = pow(10.0,true_params['log_tau'])
			true_age = pow(10.0,true_params['log_age'])
			true_mass = pow(10.0,true_params['log_mass'])
			true_t0 = 0
			true_alpha = 0
			true_beta = 0
			if sfh_form==2 or sfh_form==3:
				true_t0 = pow(10.0,true_params['log_t0'])
			if sfh_form==4:
				true_alpha = pow(10.0,true_params['log_alpha'])
				true_beta = pow(10.0,true_params['log_beta'])

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

