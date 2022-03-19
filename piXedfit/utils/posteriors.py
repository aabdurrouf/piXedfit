import numpy as np 
import sys, os
import math
import operator
from astropy.io import fits
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from scipy.interpolate import interp1d
from decimal import *
from decimal import Decimal, Context
from functools import reduce

__all__ = ["gauss_prob", "gauss_ln_prob", "student_t_prob", "model_leastnorm",  "calc_chi2", "calc_modprob_leastnorm_gauss_reduced", "calc_modprob_leastnorm_gauss", "plot_triangle_posteriors"]


def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def gauss_prob(obs_fluxes,obs_flux_err,mod_fluxes):
	d = np.asarray(obs_fluxes)
	derr = np.asarray(obs_flux_err)
	m = np.asarray(mod_fluxes)

	idx_excld = np.where((np.isnan(d)==True) | (np.isinf(d)==True) | (np.isnan(derr)==True) | (np.isinf(derr)==True))
	d = np.delete(d, idx_excld[0])
	derr = np.delete(derr, idx_excld[0])
	m = np.delete(m, idx_excld[0])

	data = np.exp(-0.5*(d-m)*(d-m)/derr/derr)/derr/math.sqrt(2.0*math.pi)
	return prod(data)

def gauss_prob_reduced(obs_fluxes,obs_flux_err,mod_fluxes):
	d = np.asarray(obs_fluxes)
	derr = np.asarray(obs_flux_err)
	m = np.asarray(mod_fluxes)

	idx_excld = np.where((np.isnan(d)==True) | (np.isinf(d)==True) | (np.isnan(derr)==True) | (np.isinf(derr)==True))
	d = np.delete(d, idx_excld[0])
	derr = np.delete(derr, idx_excld[0])
	m = np.delete(m, idx_excld[0])

	chi2 = np.sum((m-d)*(m-d)/derr/derr)
	prob = np.exp(-0.5*chi2)

	return prob

def gauss_ln_prob(obs_fluxes,obs_flux_err,mod_fluxes):
	d = np.asarray(obs_fluxes)
	derr = np.asarray(obs_flux_err)
	m = np.asarray(mod_fluxes)

	idx_excld = np.where((np.isnan(d)==True) | (np.isinf(d)==True) | (np.isnan(derr)==True) | (np.isinf(derr)==True))
	d = np.delete(d, idx_excld[0])
	derr = np.delete(derr, idx_excld[0])
	m = np.delete(m, idx_excld[0])

	data = np.exp(-0.5*(d-m)*(d-m)/derr/derr)/derr/math.sqrt(2.0*math.pi)
	ln_data = np.log(data)
	ln_prob = np.sum(ln_data)
	return ln_prob

def student_t_prob(dof,t):
	"""A function for calculating probability/likelihood based on Student's t distribution

	:param dof:
		Degree of freedom in the Student's t function

	:param t:
		Argument in the Student's t function
	"""
	idx_excld = np.where((np.isnan(t)==True) | (np.isinf(t)==True))
	t = np.delete(t, idx_excld[0])

	base = 1.0 + (t*t/dof)
	power = -0.5*(dof+1.0)
	data = math.gamma(0.5*(dof+1.0))*np.power(base,power)/math.sqrt(dof*math.pi)/math.gamma(0.5*dof)
	
	return prod(data)

def model_leastnorm(obs_fluxes,obs_flux_err,mod_fluxes):
	"""A function for calculating model normalization from chi-square minimization

	:param obs_fluxes:
		Observed multiband photometric fluxes

	:param obs_flux_err:
		Observed multiband photometric flux uncertainties

	:param mod_fluxes:
		Model multiband photometric fluxes  
	"""
	d = np.asarray(obs_fluxes)
	derr = np.asarray(obs_flux_err)
	m = np.asarray(mod_fluxes)

	idx_excld = np.where((np.isnan(d)==True) | (np.isinf(d)==True) | (np.isnan(derr)==True) | (np.isinf(derr)==True))
	d = np.delete(d, idx_excld[0])
	derr = np.delete(derr, idx_excld[0])
	m = np.delete(m, idx_excld[0])
	
	u = np.sum(d*m/derr/derr)
	l = np.sum(m*m/derr/derr) 

	norm0 = u/l
	return norm0

def calc_chi2(obs_fluxes,obs_flux_err,mod_fluxes):
	"""A function for calculting chi-square 

	:param obs_fluxes:
		Observed multiband photometric fluxes

	:param obs_flux_err:
		Observed multiband photometric flux uncertainties

	:param mod_fluxes:
		Model multiband photometric fluxes
	"""
	d = np.asarray(obs_fluxes)
	derr = np.asarray(obs_flux_err)
	m = np.asarray(mod_fluxes)

	idx_excld = np.where((np.isnan(d)==True) | (np.isinf(d)==True) | (np.isnan(derr)==True) | (np.isinf(derr)==True))
	d = np.delete(d, idx_excld[0])
	derr = np.delete(derr, idx_excld[0])
	m = np.delete(m, idx_excld[0])

	chi2 = np.sum((m-d)*(m-d)/derr/derr)
	return chi2

### define function to calculate model probability and chi-square to be used in initial fitting function:
def calc_modprob_leastnorm_gauss_reduced(obs_fluxes,obs_flux_err,mod_fluxes):
	"""A function for calculating model probability, chi-square, and normalization. 
	To be used in the initial fitting in MCMC fitting.

	"""
	d = np.asarray(obs_fluxes)
	derr = np.asarray(obs_flux_err)
	m = np.asarray(mod_fluxes)

	idx_excld = np.where((np.isnan(d)==True) | (np.isinf(d)==True) | (np.isnan(derr)==True) | (np.isinf(derr)==True))
	d = np.delete(d, idx_excld[0])
	derr = np.delete(derr, idx_excld[0])
	m = np.delete(m, idx_excld[0])

	norm0 = model_leastnorm(d,derr,m)
	mod = norm0*m
	chi2 = np.sum((mod - d)*(mod - d)/derr/derr)
	prob0 = math.exp(-0.5*chi2)

	return prob0,chi2,norm0

### define function to calculate model chi-square:
def calc_modchi2_leastnorm(obs_fluxes,obs_flux_err,mod_fluxes):
	"""A function for calculating model chi-square, and normalization. 
	"""
	d = np.asarray(obs_fluxes)
	derr = np.asarray(obs_flux_err)
	m = np.asarray(mod_fluxes)

	idx_excld = np.where((np.isnan(d)==True) | (np.isinf(d)==True) | (np.isnan(derr)==True) | (np.isinf(derr)==True))
	d = np.delete(d, idx_excld[0])
	derr = np.delete(derr, idx_excld[0])
	m = np.delete(m, idx_excld[0])

	norm0 = model_leastnorm(d,derr,m)
	mod = norm0*m
	chi2 = np.sum((mod - d)*(mod - d)/derr/derr)

	return chi2


def calc_modprob_leastnorm_gauss(obs_fluxes,obs_flux_err,mod_fluxes):
	"""A function for calculating model probability, chi-square, and normalization. 
	To be used in the initial fitting in MCMC fitting.

	"""
	norm0 = model_leastnorm(obs_fluxes,obs_flux_err,mod_fluxes)
	mod = norm0*mod_fluxes
	chi2 = np.sum((mod - obs_fluxes)*(mod - obs_fluxes)/obs_flux_err/obs_flux_err)
	data = np.exp(-0.5*(obs_fluxes-mod)*(obs_fluxes-mod)/obs_flux_err/obs_flux_err)/obs_flux_err/math.sqrt(2.0*math.pi)
	prob0 = prod(data)
	return prob0,chi2,norm0


def linear_interpolation(x,ref_x,ref_y):
	"""A function for linear interpolation
	reference: (x0,y0) and (x1,y1)
	ref_x=[x0,x1] and ref_y=[y0,y1]
	"""
	y = ref_y[0] + ((x-ref_x[0])*(ref_y[1]-ref_y[0])/(ref_x[1]-ref_x[0]))
	return y

def get_margin(sampler):
	"""A function to define margin from one array of sampler chains
	"""
	val = sampler
	array_samp = val[np.logical_not(np.isnan(val))]
	sigma = np.percentile(array_samp,84) - np.percentile(array_samp,16)
	min_margin = np.percentile(array_samp,50) - 1.5*sigma
	max_margin = np.percentile(array_samp,50) + 1.5*sigma
	return min_margin,max_margin 

def get_1D_PDF_posterior(sampler,min_sampler,max_sampler,nbins):
	"""A function to derive a 1D PDF of posterior probability distribution.
	"""
	nsamplers = len(sampler)
	del_val = (max_sampler-min_sampler)/nbins
	grid_min = np.zeros(nbins)
	grid_max = np.zeros(nbins)
	grid_prob = np.zeros(nbins)
	for ii in range(0,nbins):
		min_val = min_sampler + int(ii)*del_val
		max_val = min_sampler + (int(ii)+1.0)*del_val
		tot_prob = 0
		for xx in sampler:
			if min_val<=float(xx)<max_val:
				tot_prob = tot_prob + 1.0
		grid_min[int(ii)] = min_val
		grid_max[int(ii)] = max_val
		grid_prob[int(ii)] = float(tot_prob/nsamplers)
	return grid_min,grid_max,grid_prob

def construct_1D_histogram(grid_min,grid_max,grid_prob):
	"""A function to construct step histogram from an 1D PDF
	"""
	nbins = len(grid_min)
	hist_val = []
	hist_prob = []
	hist_val.append(grid_min[0])
	hist_prob.append(0)
	for ii in range(0,nbins):
		hist_val.append(grid_min[int(ii)])
		hist_prob.append(grid_prob[int(ii)])
		hist_val.append(grid_max[int(ii)])
		hist_prob.append(grid_prob[int(ii)])
	hist_val.append(grid_max[int(ii)-1])
	hist_prob.append(0)
	return hist_val,hist_prob   

def get_2D_PDF_posterior(sampler1,min_sampler1,max_sampler1,nbins1,sampler2,min_sampler2,max_sampler2,nbins2):
	"""A function to derive 2D posterior probability distribution
	"""
	nsamplers = len(sampler1)
	del_val1 = (max_sampler1-min_sampler1)/nbins1
	min_grid_sampler1 = np.zeros(nbins1)
	max_grid_sampler1 = np.zeros(nbins1)
	for ii in range(0,nbins1):
		min_grid_sampler1[int(ii)] = min_sampler1 + int(ii)*del_val1
		max_grid_sampler1[int(ii)] = min_sampler1 + (int(ii)+1.0)*del_val1

	del_val2 = (max_sampler2-min_sampler2)/nbins2
	min_grid_sampler2 = np.zeros(nbins2)
	max_grid_sampler2 = np.zeros(nbins2)
	for ii in range(0,nbins2):
		min_grid_sampler2[int(ii)] = min_sampler2 + int(ii)*del_val2
		max_grid_sampler2[int(ii)] = min_sampler2 + (int(ii)+1.0)*del_val2

	prob_2D = np.zeros((nbins1,nbins2))
	for ii in range(0,nsamplers):
		status1 = 0
		status2 = 0
		for yy in range(0,nbins1):
			if min_grid_sampler1[int(yy)]<=sampler1[int(ii)]<max_grid_sampler1[int(yy)]:
				idx_y = int(yy)
				status1 = 1
				break
		for xx in range(0,nbins2):
			if min_grid_sampler2[int(xx)]<=sampler2[int(ii)]<max_grid_sampler2[int(xx)]:
				idx_x = int(xx)
				status2 = 1
				break
		if status1==1 and status2==1:
			temp = prob_2D[int(idx_y)][int(idx_x)]
			prob_2D[int(idx_y)][int(idx_x)] = temp + float(1.0/nsamplers)
	return prob_2D

def calc_mode(sampler_chains0):
	"""A function for calculating mode from a sampler chains
	"""
	# exclude nan values:
	x = sampler_chains0
	x1 = x[np.logical_not(np.isnan(x))]
	sampler_chains = x1[np.logical_not(np.isinf(x1))]
	nchains = len(sampler_chains)
	if len(sampler_chains) > 0:
		nbins = 20
		min_val = np.percentile(sampler_chains, 5)
		max_val = np.percentile(sampler_chains, 95)

		del_val = (max_val-min_val)/nbins

		grid_val = np.zeros(nbins)
		grid_prob = np.zeros(nbins)
		for ii in range(0,nbins):
			x0 = min_val + (int(ii)*del_val)
			x1 = min_val + ((int(ii)+1.0)*del_val)
			count = 0
			for xx in sampler_chains:
				if x0<=float(xx)<x1:
					count = count + 1.0
			grid_val[int(ii)] = min_val + ((int(ii)+0.5)*del_val)
			grid_prob[int(ii)] = count/nchains
		# interpolate to get finer sampling of the PDF:
		f = interp1d(grid_val, grid_prob, kind='cubic')
		val_new = np.linspace(min(grid_val),max(grid_val),60, endpoint=True)
		prob_new = f(val_new)
		# get maxima:
		idx, max_val = max(enumerate(prob_new), key=operator.itemgetter(1))
		mode = val_new[int(idx)]
	else:
		mode = -99.0
	return mode

def calc_mode_PDF(val_min,val_max,grid_val,grid_prob):
	ngrids = len(grid_val)
	# interpolate to get finer sampling of the PDF:
	f = interp1d(grid_val, grid_prob, kind='cubic', bounds_error=False, fill_value='extrapolate')
	val_new = np.linspace(val_min,val_max,3*ngrids, endpoint=True)
	prob_new = f(val_new)
	# get maxima:
	idx, max_val = max(enumerate(prob_new), key=operator.itemgetter(1))
	mode = val_new[int(idx)]
	return mode

# function to make float be with 2 decimal points:
def change_2decimal(x):
	TWOPLACES = Decimal(10)**-2
	ndata = len(x)
	x_new = np.zeros(ndata)
	for ii in range(0,ndata):
		str_temp = '%lf' % x[int(ii)]
		if math.isnan(float(str_temp))==False and math.isinf(float(str_temp))==False:
			x_new[int(ii)] = Decimal(str_temp).quantize(TWOPLACES)
		else:
			x_new[int(ii)] = -999.99
	return x_new

# function to calculate percentiles from a PDF:
def percentile(PDF_values,PDF_prob,perc_idx):  # perc_idx=16 for 16th percentile
	nbins = len(PDF_values)
	# normalize the PDF:
	tot_prob = np.sum(PDF_prob)
	for ii in range(0,nbins):
		temp = PDF_prob[int(ii)]
		PDF_prob[int(ii)] = temp/tot_prob
	# calculate cumulative PDF:
	cumul_PDF_prob = np.zeros(nbins)
	tot_prob = 0
	for ii in range(0,nbins):
		tot_prob = tot_prob + PDF_prob[int(ii)]
		cumul_PDF_prob[int(ii)] = tot_prob
	# calculate the percentile:
	perc_idx = perc_idx/100.0
	perc = 0
	for ii in range(0,nbins-1):
		if cumul_PDF_prob[int(ii)]<perc_idx and cumul_PDF_prob[int(ii)+1]>perc_idx:
			perc = PDF_values[int(ii)] + (perc_idx-cumul_PDF_prob[int(ii)])*(PDF_values[int(ii)+1]-PDF_values[int(ii)])/(cumul_PDF_prob[int(ii)+1]-cumul_PDF_prob[int(ii)])
	return perc

# define function to make line histogram plot
def plot_line_histogram(f,nbins,hist_prob,x_min,x_max,perc_16,perc_50,perc_84,mode,mean,true):
	plt.setp(f.get_xticklabels(), visible=False)
	plt.setp(f.get_yticklabels(), visible=False)

	max_y = max(hist_prob)*1.1

	# plot step histogram
	x_temp = np.linspace(-0.5,nbins-0.5,nbins+1)
	x = []
	for xx in x_temp:
		x.append(float(xx))
		x.append(float(xx))
	plt.plot(x,hist_prob,linewidth=3,color='black')
	plt.ylim(0,max_y)

	# convert perc-16, median, perc-84, mode, mean, and true into the coordinate of the plane
	ref_x = np.zeros(2)
	ref_x[0] = x_min
	ref_x[1] = x_max
	ref_y = np.zeros(2)
	ref_y[0] = -0.5
	ref_y[1] = nbins-0.5
	perc16_conv = linear_interpolation(perc_16,ref_x,ref_y)
	med_conv = linear_interpolation(perc_50,ref_x,ref_y)
	perc84_conv = linear_interpolation(perc_84,ref_x,ref_y)
	if mode != -99.0:
		mode_conv = linear_interpolation(mode,ref_x,ref_y)
	if mean != -99.0:
		mean_conv = linear_interpolation(mean,ref_x,ref_y)
	if true != -99.0:
		true_conv = linear_interpolation(true,ref_x,ref_y)

	# plot vertical gray shaded area  
	trans = transforms.blended_transform_factory(f.transData, f.transAxes)
	rect = patches.Rectangle((perc16_conv ,0), width=perc84_conv-perc16_conv, height=1,transform=trans, color='gray',alpha=0.5)
	f.add_patch(rect)
	# plot median
	plot_y = np.linspace(0,max_y,10)
	plot_x = plot_y - plot_y + med_conv
	plt.plot(plot_x,plot_y,color='black',linestyle='--',lw=2)
	# plot mode
	if mode != -99.0:
		plot_x = plot_y - plot_y + mode_conv
		plt.plot(plot_x,plot_y,color='blue',linestyle='--',lw=2)
	# plot mean
	if mean != -99.0:
		plot_x = plot_y - plot_y + mean_conv
		plt.plot(plot_x,plot_y,color='green',linestyle='--',lw=2)
	# plot true
	if true != -99.0:
		plot_x = plot_y - plot_y + true_conv
		plt.plot(plot_x,plot_y,color='red',lw=2)


# function to plot 2D posterior probability density map:
def plot_2D_posteriors(nbins_x,nbins_y,min_x,max_x,min_y,max_y,array2D_prob):
	plt.xlim(-0.5,nbins_x-0.5)
	plt.ylim(-0.5,nbins_y-0.5)
	xtick = np.linspace(-0.5,nbins_x-0.5,6)
	plt.xticks(xtick)
	real_x = np.linspace(min_x,max_x,6)
	plt.gca().set_xticklabels(change_2decimal(real_x))

	ytick = np.linspace(-0.5,nbins_y-0.5,6)
	plt.yticks(ytick)
	real_y = np.linspace(min_y,max_y,6)
	plt.gca().set_yticklabels(change_2decimal(real_y))
	im = plt.imshow(array2D_prob, interpolation='bicubic',cmap=cm.Set1_r,origin='lower')


def calculate_bestfit_parameters(samplers_param):
	""" 
	## Calculate percentiles (16,50,84) of model parameters from the sampler chains
	## Calculate posterior means of SFR and SM and their uncertainties
	## samplers_param[idx-param][idx-model]
	## perc_params[idx-param][idx-perc(0,1,2)]
	"""
	nparams = len(samplers_param)
	nchains = len(samplers_param[0])
	perc_params = np.zeros((nparams,3))
	for pp in range(0,nparams):
		perc_params[int(pp)][0] = np.percentile(samplers_param[int(pp)], 16)
		perc_params[int(pp)][1] = np.percentile(samplers_param[int(pp)], 50)
		perc_params[int(pp)][2] = np.percentile(samplers_param[int(pp)], 84)

	return perc_params


def plot_triangle_posteriors(param_samplers=[],label_params=[],true_params=[],post_mean_flag=[],post_mode_flag=[],params_ranges=[],
	nbins=12,fontsize_label=20,fontsize_tick=14,output_name='corner.png'):
	"""A function for creating corner/triangle plot for posterior probability distribution of model parameters

	:param param_samplers:
		2D array containing sampler chains from the MCMC fitting. It has structure as: param_samplers[idx-param][idx-sampler chain]

	:param label_params:
		1D array of string to be used for labeling each parameter in the corner plot

	:param true_params:
		1D array of true values of the parameters, in case the true values are exist and are going to be used in the plot.

	:param post_mean_flag:
		1D array of Flag stating whether to plot (1) mean posterior value or not (0).

	:param post_mode_flag:
		1D arrar of Flag stating whether to plot (1) mode posterior value or not (0).

	:param params_ranges:
		2D array of prior ranges of the parameters. The structure: params_ranges[idx-param]=[min_margin,max_margin].

	:param nbins (default: 12):
		Number of bins in each parameter side in the calculation of 1D and 2D PDFs.

	:param fontsize_label (default: 20):
		Fontsize for the labels

	:param fontsize_tick (default: 14):
		Fontsize for the ticks

	:param output_name (default: 'corner.png'):
		Name for the output plot.
	"""

	nparams = len(param_samplers)
	print ("Plotting posteriors probability distributions")
	print ("=> Number of parameters: %d" % nparams)

	
	fig1 = plt.figure(figsize=(18,18))

	idx = 0
	for p1 in range(0,nparams):       # y-axis
		for p2 in range(0,nparams):   # x-axis
			idx = idx + 1
			if p2 <= p1:
				f1 = plt.subplot(nparams,nparams,int(idx))
            	# make 1D PDF:
				if p1 == p2:
					val = param_samplers[p1]
					array_samp = val[np.logical_not(np.isnan(val))]
					perc_16 = np.percentile(array_samp,16)
					perc_50 = np.percentile(array_samp,50)
					perc_84 = np.percentile(array_samp,84)
					min_margin,max_margin = get_margin(param_samplers[p1])
					x_min = min_margin
					x_max = max_margin

					# use the prior ranges (in the fitting) if the margin exceed beyond it
					if x_min < params_ranges[p1][0] and params_ranges[p1][0] != -99.0:
						x_min = params_ranges[p1][0]
					if x_max > params_ranges[p1][1] and params_ranges[p1][1] != -99.0:
						x_max = params_ranges[p1][1]

					# mean
					if post_mean_flag[p1]==0:
						mean = -99.0
					elif post_mean_flag[p1]==1:
						mean = np.mean(param_samplers[p1])

					# mode
					if post_mode_flag[p1]==0:
						mode = - 99.0
					elif post_mode_flag[p1]==1:
						mode = calc_mode(param_samplers[p1])

					grid_min,grid_max,grid_prob = get_1D_PDF_posterior(param_samplers[p1],x_min,x_max,nbins)
					hist_val,hist_prob = construct_1D_histogram(grid_min,grid_max,grid_prob)
					plot_line_histogram(f1,nbins,hist_prob,x_min,x_max,perc_16,perc_50,perc_84,mode,mean,true_params[p1])

					if int(p1) == nparams-1:
						plt.xlabel(r'%s' % label_params[int(p1)], fontsize=int(fontsize_label))
						plt.setp(f1.get_yticklabels(), visible=False)
						plt.setp(f1.get_xticklabels(), visible=True, fontsize=int(fontsize_tick))
						xtick = np.linspace(-0.5,nbins-0.5,6)
						plt.xticks(xtick)
						real_x = np.linspace(x_min,x_max,6)
						plt.gca().set_xticklabels(change_2decimal(real_x))
						plt.xticks(rotation='vertical')

            	# make 2D PDF:
				else:
					min_margin,max_margin = get_margin(param_samplers[int(p1)])
					y_min = min_margin
					y_max = max_margin
					# use the prior ranges (in the fitting) if the margin exceed beyond it
					if y_min < params_ranges[int(p1)][0] and params_ranges[int(p1)][0] != -99.0:
						y_min = params_ranges[int(p1)][0]
					if y_max > params_ranges[int(p1)][1] and params_ranges[int(p1)][1] != -99.0:
						y_max = params_ranges[int(p1)][1]
			
					min_margin,max_margin = get_margin(param_samplers[int(p2)])
					x_min = min_margin
					x_max = max_margin

					# use the prior ranges (in the fitting) if the margin exceed beyond it
					if x_min < params_ranges[int(p2)][0] and params_ranges[int(p2)][0] != -99.0:
						x_min = params_ranges[int(p2)][0]
					if x_max > params_ranges[int(p2)][1] and params_ranges[int(p2)][1] != -99.0:
						x_max = params_ranges[int(p2)][1]

					array2D_prob = get_2D_PDF_posterior(param_samplers[p1],y_min,y_max,nbins,param_samplers[p2],x_min,x_max,nbins)
					plot_2D_posteriors(nbins,nbins,x_min,x_max,y_min,y_max,array2D_prob)

					if int(p2)==0 and int(p1)!=nparams-1:
						plt.ylabel(r'%s' % label_params[int(p1)],fontsize=int(fontsize_label))
						plt.setp(f1.get_xticklabels(), visible=False)
						plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
						plt.xticks(rotation='vertical')
					elif int(p2)==0 and int(p1)==nparams-1:
						plt.xlabel(r'%s' % label_params[int(p2)],fontsize=int(fontsize_label))
						plt.ylabel(r'%s' % label_params[int(p1)],fontsize=int(fontsize_label))
						plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))
						plt.setp(f1.get_yticklabels(), fontsize=int(fontsize_tick))
						plt.xticks(rotation='vertical')
					elif int(p1)==nparams-1 and int(p2)!=0:
						plt.xlabel(r'%s' % label_params[int(p2)],fontsize=int(fontsize_label))
						plt.setp(f1.get_yticklabels(), visible=False)
						plt.setp(f1.get_xticklabels(), fontsize=int(fontsize_tick))
						plt.xticks(rotation='vertical')
					else:
						plt.setp(f1.get_yticklabels(), visible=False)
						plt.setp(f1.get_xticklabels(), visible=False)

	plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.98, wspace=0.05, hspace=0.05)
	plt.savefig(output_name)


