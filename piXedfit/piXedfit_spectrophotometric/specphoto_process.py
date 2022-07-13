import numpy as np 
import sys, os
import random
from astropy.io import fits
from astropy.wcs import WCS
from operator import itemgetter 
from astropy.convolution import convolve_fft, Gaussian1DKernel
from reproject import reproject_exact
from photutils.psf.matching import resize_psf
from ..piXedfit_images.images_utils import get_largest_FWHM_PSF, k_lmbd_Fitz1986_LMC

try:
	global PIXEDFIT_HOME
	PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_spectrophotometric/'
	temp_dir = PIXEDFIT_HOME+'/data/temp/'
except:
	print ("PIXEDFIT_HOME should be included in your PATH!")


__all__ = ["match_imgifs_spatial", "match_imgifs_spectral"]


def specphoto_califagalexsdss2masswise(photo_fluxmap, califa_file, spec_smoothing=False, kernel_sigma=2.6, 
	nproc=10, name_out_fits=None):
	"""Function for matching spatially (on pixel level) between IFS data cube from CALIFA and the multiwavelength imaging 
	data (12 bands from GALEX, SDSS, 2MASS, and WISE). 

	:param photo_fluxmap:
		Input 3D data cube of photometry. This should have the same format as the output of :func:`piXedfit.piXedfit_images.images_processing.flux_map`.

	:param califa_file:
		Input CALIFA data cube.

	:param spec_smoothing: (default: False)
		If True, spectrum of each pixel will be smoothed by convolving it with a Gaussian kernel with a standard deviation given by the input kernel_sigma.  

	:param kernel_sigma: (default: 2.6)
		Standard deviation of the kernel to be convolved with the spectrum of each pixel.

	:param nproc: (default: 10)
		Number of cores to be used for the calculation.

	:param name_out_fits:
		Name of output FITS file.
	"""

	# make configuration file
	name_config = "config_file%d.dat" % (random.randint(0,10000))
	file_out = open(name_config,"w")
	file_out.write("photo_fluxmap %s\n" % photo_fluxmap)
	file_out.write("califa_file %s\n" % califa_file)
	if spec_smoothing==True or spec_smoothing==1:
		spec_smoothing1 = 1
	elif spec_smoothing==False or spec_smoothing==0:
		spec_smoothing1 = 0
	else:
		print ("Not known input for spec_smoothing!")
		sys.exit()
	file_out.write("spec_smoothing %d\n" % spec_smoothing1)
	file_out.write("kernel_sigma %lf\n" % kernel_sigma)
	if name_out_fits==None:
		temp1 = photo_fluxmap.replace('.fits','')
		temp2 = califa_file.replace('.fits','')
		name_out_fits = "specphoto_%s_%s.fits" % (temp1,temp2)
	file_out.write("name_out_fits %s\n" % name_out_fits)
	file_out.close()

	os.system('mv %s %s' % (name_config,temp_dir))

	# get number of wavelength points
	cube = fits.open(califa_file)
	min_wave = float(cube[0].header['CRVAL3'])
	del_wave = float(cube[0].header['CDELT3'])
	nwaves = int(cube[0].header['NAXIS3'])
	max_wave = min_wave + (nwaves-1)*del_wave
	wave = np.linspace(min_wave,max_wave,nwaves)
	cube.close()

	# determine number of cores
	if nproc>10:
		nwaves = len(wave)
		modulo = []
		nproc0 = []
		for ii in range(int(nproc),int(nproc)-4,-1):
			mod0 = nwaves % ii
			modulo.append(mod0)
			nproc0.append(ii)
		idx0, min_val = min(enumerate(np.asarray(modulo)), key=itemgetter(1))
		nproc_new = int(nproc0[idx0])
	else:
		nproc_new = nproc

	os.system("mpirun -n %d python %s./sp_clf.py %s" % (nproc_new,CODE_dir,name_config))

	return name_out_fits


def specphoto_mangagalexsdss2masswise(photo_fluxmap, manga_file, spec_smoothing=False, kernel_sigma=3.5, 
	nproc=10, name_out_fits=None):
	
	"""Function for matching (spatially on pixel scales) between IFS data cube from MaNGA and the multiwavelength imaging 
	data (12 bands from GALEX, SDSS, 2MASS, and WISE). 

	:param photo_fluxmap:
		Input 3D data cube of photometry. This should have the same format as the output of :func:`piXedfit.piXedfit_images.images_processing.flux_map`.

	:param manga_file:
		Input MaNGA data cube.

	:param spec_smoothing: (default: False)
		If True, spectrum of each pixel will be smoothed by convolving it with a Gaussian kernel with a standard deviation given by the input kernel_sigma. 

	:param kernel_sigma: (default: 3.5)
		Standard deviation of the kernel to be convolved with the spectrum of each pixel.

	:param nproc: (default: 10)
		Number of cores to be used for the calculation.

	:param name_out_fits:
		Name of output FITS file.
	"""

	# make configuration file
	name_config = "config_file%d.dat" % (random.randint(0,10000))
	file_out = open(name_config,"w")
	file_out.write("photo_fluxmap %s\n" % photo_fluxmap)
	file_out.write("manga_file %s\n" % manga_file)
	if spec_smoothing==True or spec_smoothing==1:
		spec_smoothing1 = 1
	elif spec_smoothing==False or spec_smoothing==0:
		spec_smoothing1 = 0
	else:
		print ("Not known input for spec_smoothing!")
		sys.exit()
	file_out.write("spec_smoothing %d\n" % spec_smoothing1)
	file_out.write("kernel_sigma %lf\n" % kernel_sigma)
	if name_out_fits==None:
		temp1 = photo_fluxmap.replace('.fits','')
		temp2 = manga_file.replace('.fits','')
		name_out_fits = "specphoto_%s_%s.fits" % (temp1,temp2)
	file_out.write("name_out_fits %s\n" % name_out_fits)
	file_out.close()

	os.system('mv %s %s' % (name_config,temp_dir))

	## open MaNGA IFS data
	cube = fits.open(manga_file)
	wave = cube['WAVE'].data
	cube.close()

	# determine number of cores
	if nproc>10:
		nwaves = len(wave)
		modulo = []
		nproc0 = []
		for ii in range(int(nproc),int(nproc)-4,-1):
			mod0 = nwaves % ii
			modulo.append(mod0)
			nproc0.append(ii)
		idx0, min_val = min(enumerate(np.asarray(modulo)), key=itemgetter(1))
		nproc_new = int(nproc0[idx0])
	else:
		nproc_new = nproc

	os.system("mpirun -n %d python %s./sp_mga.py %s" % (nproc_new,CODE_dir,name_config))

	return name_out_fits


def match_imgifs_spatial(photo_fluxmap,ifs_data,ifs_survey='manga',spec_smoothing=False,kernel_sigma=3.5,
	nproc=10, name_out_fits=None):
	
	"""Function for matching spatially (on pixel level) between IFS data cube and a post-processed multiwavelength imaging 
	data cube. The current version of piXedfit only can process IFS data from CALIFA and MaNGA surveys.  

	:param photo_fluxmap:
		Input 3D data cube of photometry. This data cube is an output of function :func:`flux_map` in the :class:`images_processing` class.

	:param ifs_data:
		Integral Field Spectroscopy (IFS) data cube.

	:param ifs_survey:
		The survey from which the IFS data cube is taken. Options are: 'manga' and 'califa'. 

	:param spec_smoothing:
		If True, spectrum of each pixel will be smoothed by convolving it with a Gaussian kernel with a standard deviation given by the input kernel_sigma. 

	:param kernel_sigma:
		Standard deviation of the kernel to be convolved with the spectrum of each pixel.

	:param nproc:
		Number of cores to be used for the calculation.

	:param name_out_fits:
		Desired name for the output FITS file.
	"""

	if ifs_survey=='manga':
		specphoto_mangagalexsdss2masswise(photo_fluxmap,ifs_data,spec_smoothing=spec_smoothing,
										kernel_sigma=kernel_sigma,nproc=nproc,name_out_fits=name_out_fits)
	elif ifs_survey=='califa':
		specphoto_califagalexsdss2masswise(photo_fluxmap,ifs_data,spec_smoothing=spec_smoothing,
										kernel_sigma=kernel_sigma,nproc=nproc,name_out_fits=name_out_fits)
	else:
		print ("The inputted ifs_source is not recognized!")
		sys.exit()


def match_imgifs_spectral(specphoto_file, models_spec=None, nproc=10, del_wave_nebem=10.0, cosmo=0, 
	H0=70.0, Om0=0.3, name_out_fits=None):

	"""Function for correcting wavelength-dependent mismatch between the IFS spectra and the photometric SEDs (on pixel level) 
	in the spectrophotometric data cube that is produced using the function :func:`piXedfit.piXedfit_spectrophotometric.match_imgifs_spatial`. 
	
	:param specphoto_file:
		Input spectrophotometric data cube that is produced using :func:`piXedfit.piXedfit_spectrophotometric.match_imgifs_spatial`.

	:param models_spec:
		Set of model spectra at rest-frame produced using :func:`piXedfit.piXedfit_model.save_models_rest_spec`. The file is in the HDF5 format. 
		For more information on how to produce this set of models, please see the description :ref:`here <gen_models_seds>`.
		This set of models only need to be produced once and then it can be used for all galaxies in a sample. 
		If models_spec=None, a default file is then called from piXedfit/data/mod ($PIXEDFIT_HOME/data/mod). 
		However, this file is not available in that directory at first piXedfit is installed, instead user need to download it 
		from `this link <https://drive.google.com/drive/folders/1YjZGg97dPT8S95NJmO5tiFH9jWhbxuVy?usp=sharing>`_ and put it on that 
		directory in the local machine.   

	:param nproc:
		Number of cores to be used for calculation.

	:param del_wave_nebem: (default: 10.0 Angstrom).
		The range (+/-) around all emission lines in the model spectra that will be removed in producing spectral continuum, 
		which will be used as reference for correcting the wavelength-dependent mismatch between the IFS spectra and photometric SEDs.    

	:param cosmo:
		Choices for the cosmology. Options are: (a)'flat_LCDM' or 0, (b)'WMAP5' or 1, (c)'WMAP7' or 2, (d)'WMAP9' or 3, (e)'Planck13' or 4, (f)'Planck15' or 5.
		These options are the same to that available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0:
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0:
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param name_out_fits:
		Desired name for the output FITS file. If None, a generic name will be used.
	"""

	# make configuration file
	name_config = "config_file%d.dat" % (random.randint(0,10000))
	file_out = open(name_config,"w")
	file_out.write("specphoto_file %s\n" % specphoto_file)
	if models_spec == None:
		models_spec = 'none'
	file_out.write("models_spec %s\n" % models_spec)
	file_out.write("del_wave_nebem %lf\n" % del_wave_nebem)

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

	if name_out_fits==None:
		name_out_fits= "corr_%s" % specphoto_file
	file_out.write("name_out_fits %s\n" % name_out_fits)
	file_out.close()

	os.system('mv %s %s' % (name_config,temp_dir))
	os.system("mpirun -n %d python %s./mtch_sph_fnal.py %s" % (nproc,CODE_dir,name_config))

	return name_out_fits





