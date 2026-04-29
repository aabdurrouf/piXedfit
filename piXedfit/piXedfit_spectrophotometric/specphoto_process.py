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
from .specphoto_utils import *

try:
	global PIXEDFIT_HOME
	PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_spectrophotometric/'
	temp_dir = PIXEDFIT_HOME+'/data/temp/'
except:
	print ("PIXEDFIT_HOME should be included in your PATH!")


__all__ = ["match_imgifs_spatial", "match_imgifs_spectral", "combine_nirspec_ifu_and_images"]


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
		from `this link <https://drive.google.com/drive/folders/1Fvl42e_LNWLYhKabDS1ew6wTQjeopcc6?usp=sharing>`_ and put it on that 
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


def combine_nirspec_ifu_and_images(nirspec_ifu, disperser_filter, gal_ra, gal_dec, gal_z, filters, sci_images, var_images=None, dir_images='./', 
                                   nirspec_psf_cubes=None, psf_images=None, nirspec_unit='MJy/sr', image_unit='MJy/sr', nirspec_scale=None, 
                                   image_scale=None, kernel_window='cosine_bell', kernel_window_arg=1.0, del_wave_resamp=20.0, name_out_fits=None, 
                                   flag_psfmatch=0, flag_reproject=0, ncpu=5, del_wave_nebem=20.0, poly_order=3, verbose=False):
    """ Function for combining NIRSpec IFU with imaging data (e.g., JWST, HST)
    
    :param nirspec_ifu:
        A dictionary containing names of NIRSpec IFU data cubes, with 'disperser-filter' as keys. All the IFU data cubes should be WCS-aligned and have the same 2D size/dimension.

    :param disperser_filter:
        List of 'disperser-filter' of the input NIRSpec IFU data cubes. For example: 'g140h-f100lp', 'g235h-f170lp' 

    :param gal_ra:
        RA of the target galaxy.

    :param gal_dec:
        DEC of the target galaxy.

    :param gal_z:
        Redshift of the target galaxy.

    :param filters:
        List of photometric filters.

    :param sci_images:
        A dictionary containing input science images, with filter names as keys.

    :param var_images:
        A dictionary containing input varaiance images, with filter names as keys. Both err_images and var_images can't be empty.

    :param dir_images:
        Directory of the input images.

    :param nirspec_psf_cubes:
        A dictionary containing fits files for the wavelength-dependant model PSF of the NIRSpec IFU, with 'disperser-filter' as keys. Should have the same pixel size as that of input NIRSpec IFU data.

    :param psf_images:
        A dictionary containing PSF images of the imaging data, with filter name as keys. Should have the same pixel size as that of input images.

    :param nirspec_unit:
        Flux unit of the NIRSpec IFU data. Options are: 'MJy/sr', 'Jy', 'erg/s/cm2/A'.

    :param image_unit:
        Flux unit of the images. Options are: 'MJy/sr', 'Jy', 'erg/s/cm2/A'.

    :param nirspec_scale:
        A dictionary for scaling factor for the pixel values relative to the unit specified in nirspec_unit. 
        For example, if the image is in units of MJy, you can set nirspec_unit to 'Jy' and nirspec_scale to 1e6.
        If None, nirspec_scale is assummed to be 1.
    
    :param image_scale:
        A dictionary for scaling factor for the pixel values relative to the unit specified in img_unit. 
        For example, if the image is in units of MJy, you can set img_unit to 'Jy' and img_scale to 1e6.
        If None, image_scale is assummed to be 1.

    :param kernel_window:
        Window function to be used for calculating kerner (for PSF matching). This is the same window parameter as that in Photutils: https://photutils.readthedocs.io/en/stable/api/photutils.psf.matching.create_matching_kernel.html

    :param kernel_window_arg:

    
    """
    from astropy.wcs import WCS
    from spectres import spectres
    from scipy.interpolate import interp1d
    from astropy.convolution import convolve_fft
    
    from ..utils.filtering import get_filter_curve
    from ..utils.filtering import add_filter, remove_filter, cwave_filters
    from ..piXedfit_images import calc_pixsize, create_psf_matching_kernel, images_processing, compute_fwhm_psf, interpolate_psf_cube

    for ch in disperser_filter:
        check_disperser_filter(ch)

    add_fil = 'fil_temp'
    remove_filter(add_fil)
    
    disperser_filter = sort_by_filter(disperser_filter, ['f070lp', 'f100lp', 'f170lp', 'f290lp'])
    
    if nirspec_scale is None:
        nirspec_scale = {}
        for ch in disperser_filter:
            nirspec_scale[ch] = 1.0

    n_ifu = len(disperser_filter)

    if image_scale is None:
        image_scale = {}
        for ff in filters:
            image_scale[ff] = 1.0

    # make stack 2D image from the NIRSpec IFU data cube.
    cube = fits.open(nirspec_ifu[disperser_filter[0]])
    sum_ifu_sci = np.nansum(cube['sci'].data, axis=0) 
    sum_ifu_var = np.nansum(np.square(cube['err'].data), axis=0)
    dimy_ifu, dimx_ifu = sum_ifu_sci.shape[0], sum_ifu_sci.shape[1]
    w = WCS(naxis=2)
    w.wcs.crpix = [float(cube['sci'].header['CRPIX1']), float(cube['sci'].header['CRPIX2'])]
    w.wcs.cdelt = np.array([float(cube['sci'].header['CDELT1']), float(cube['sci'].header['CDELT2'])])
    w.wcs.crval = [float(cube['sci'].header['CRVAL1']), float(cube['sci'].header['CRVAL2'])]
    w.wcs.ctype = [cube['sci'].header['CTYPE1'], cube['sci'].header['CTYPE2']]
    w.wcs.cunit = ['deg', 'deg']
    w.wcs.pc = [[-1.0, 0.0], [ 0.0, 1.0]]
    ifu_2d_header = w.to_header()
    ifu_img_sci, ifu_img_var = 'sci_sum_ifu.fits', 'var_sum_ifu.fits'
    fits.writeto(ifu_img_sci, sum_ifu_sci, ifu_2d_header, overwrite=True)
    fits.writeto(ifu_img_var, sum_ifu_var, ifu_2d_header, overwrite=True)
    cube.close()

    if dir_images != './':
        os.system('mv %s %s' % (ifu_img_sci, dir_images))
        os.system('mv %s %s' % (ifu_img_var, dir_images))

    photo_wave = cwave_filters(filters)

    # calculate pixel sizes of imaging data
    img_pix = {}
    for ff in filters:
        img_pix[ff] = calc_pixsize(dir_images + sci_images[ff])

    if flag_psfmatch == 0:
        # determine the largest PSF 
        max_fwhm_ifu = np.zeros(n_ifu)
        for ii in range(n_ifu):
            #cube = fits.open(nirspec_psf_cubes[disperser_filter[ii]])
            #results = compute_fwhm_psf(cube['det_dist'].data[-1,:,:], error_image=None)
            #max_fwhm_ifu[ii] = results['fwhm']*float(cube['det_dist'].header['PIXELSCL'])
            #cube.close()

            # open PSF cube for this channel
            cube1 = fits.open(nirspec_psf_cubes[disperser_filter[ii]])
            cube_psf = cube1['DET_DIST'].data
            cube_psf_wave = np.zeros(cube_psf.shape[0])
            for i in range(cube_psf.shape[0]):
                cube_psf_wave[i] = cube1['det_dist'].header["WVLN%04d" % i]*1e+6
            psf_pixsize = float(cube1['det_dist'].header['PIXELSCL'])
            cube1.close()

            cube = fits.open(nirspec_ifu[disperser_filter[ii]])
            waves_um = get_wavelength_grid_nirspec(cube['sci'].header)
            cube.close()
            
            if max(waves_um) < max(cube_psf_wave):
                psf_interp = interpolate_psf_cube(cube_psf, cube_psf_wave, max(waves_um), kind='cubic')
            else:
                psf_interp = cube_psf[-1,:,:]

            results = compute_fwhm_psf(psf_interp, error_image=None)
            max_fwhm_ifu[ii] = results['fwhm']*psf_pixsize

        fwhm_photo = np.zeros(len(filters))
        for ii in range(len(filters)):
            hdu = fits.open(psf_images[filters[ii]])        
            results = compute_fwhm_psf(hdu[0].data, error_image=None)
            fwhm_photo[ii] = results['fwhm']*img_pix[filters[ii]]
            hdu.close()

        cmb_fils = disperser_filter + filters
        cmb_fwhm = max_fwhm_ifu.tolist() + fwhm_photo.tolist()
        idx_max = cmb_fwhm.index(max(cmb_fwhm))

        # get largest PSF image
        if idx_max < n_ifu:
            # largest PSF is from NIRSpec IFU
            cube = fits.open(nirspec_psf_cubes[disperser_filter[idx_max]])
            max_psf = cube['DET_DIST'].data[-1,:,:]
            max_psf_pixsize = float(cube['DET_DIST'].header['PIXELSCL'])
            max_fwhm = max_fwhm_ifu[idx_max]
            fil_max_psf = disperser_filter[idx_max]
            cube.close()
            if verbose:
                print ('PSF matching will be done to the largest PSF with FWHM = %.2f arcsec (%s)' % (max_fwhm,fil_max_psf))
        else:   
            # largest PSF is from imaging data
            max_psf = fits.getdata(psf_images[filters[idx_max-n_ifu]])
            max_psf_pixsize = img_pix[filters[idx_max-n_ifu]]
            max_fwhm = fwhm_photo[idx_max-n_ifu]
            fil_max_psf = filters[idx_max-n_ifu]
            if verbose:
                print ('PSF matching will be done to the largest PSF with FWHM = %.2f arcsec (%s)' % (max_fwhm,fil_max_psf))

        # save the largest PSF image
        name_largest_psf = 'largest_psf.fits'
        fits.writeto(name_largest_psf, max_psf, overwrite=True)

    ## in the image processing, ignore PSF matching for the IFU data cube because it will be done later as the IFU data cube is not changed by reprojection
    filter_name = add_fil
    filter_cwave = photo_wave[0] - 500
    filter_wave = np.linspace(filter_cwave-2e+2, filter_cwave+2e+2, 400)
    filter_transmission = np.zeros(len(filter_wave)) + 1
    add_filter(filter_name, filter_wave, filter_transmission, filter_cwave)

    sci_img = {}
    var_img = {}
    kernels = {}
    img_unit = {}
    img_scale = {}
    img_pixsizes = {}

    # add 2d img ifu
    filters.append(add_fil)
    sci_img[add_fil] = ifu_img_sci
    var_img[add_fil] = ifu_img_var
    kernels[add_fil] = None
    img_unit[add_fil] = nirspec_unit 
    img_scale[add_fil] = nirspec_scale
    img_pixsizes[add_fil] = calc_pixsize(dir_images+ifu_img_sci)

    for ii in range(len(filters)-1):
        sci_img[filters[ii]] = sci_images[filters[ii]]
        var_img[filters[ii]] = var_images[filters[ii]]
        img_unit[filters[ii]] = image_unit
        img_scale[filters[ii]] = image_scale
        img_pixsizes[filters[ii]] = img_pix[filters[ii]]

        if flag_psfmatch == 1:
             kernels[filters[ii]] = None
        else:
            if filters[ii]==fil_max_psf:
                kernels[filters[ii]] = None
            else:
                # calculate kernel
                init_PSF_name = psf_images[filters[ii]]
                target_PSF_name = name_largest_psf
                pixscale_init_PSF = img_pixsizes[filters[ii]]
                pixscale_target_PSF = max_psf_pixsize
                kernel = create_psf_matching_kernel(init_PSF_name, target_PSF_name, pixscale_init_PSF, pixscale_target_PSF, 
                                                        window=kernel_window, window_arg=kernel_window_arg)
                kernel_name = 'kernel_for_'+filters[ii]+'.fits'
                fits.writeto(kernel_name, kernel, overwrite=True)
                if verbose:
                    print ('produced '+kernel_name+' for PSF matching of '+filters[ii])
                kernels[filters[ii]] = kernel_name

    stamp_size = [int(dimx_ifu*2), int(dimx_ifu*2)]
    if verbose==True and flag_psfmatch==0:
        print ('perform PSF matching and reprojection for the images...')
    img_process = images_processing(filters, sci_img, var_img, gal_ra, gal_dec, dir_images=dir_images, img_unit=img_unit, img_scale=img_scale, 
                                    img_pixsizes=img_pixsizes, kernels=kernels, gal_z=gal_z, stamp_size=stamp_size, flag_psfmatch=flag_psfmatch, 
                                    flag_reproject=flag_reproject, flag_crop=0, idfil_align=0, remove_files=False, verbose=False)
    output_stamps = img_process.get_output_stamps()

    os.system('rm %s%s' % (dir_images,ifu_img_sci))
    os.system('rm %s%s' % (dir_images,ifu_img_var))

    # get NIRSpec IFU coverage region
    nirspec_region = np.zeros(sum_ifu_sci.shape)
    rows, cols = np.where(sum_ifu_var > 0.0)
    nirspec_region[rows,cols] = 1.0

    filters.remove(add_fil)
    nbands = len(filters)
    dimy, dimx = sum_ifu_sci.shape[0], sum_ifu_sci.shape[1]

    # get WCS header information
    hdu = fits.open(output_stamps['name_img_'+filters[0]])
    w = WCS(naxis=2)
    w.wcs.crpix = [float(hdu[0].header['CRPIX1']), float(hdu[0].header['CRPIX2'])]
    w.wcs.cdelt = np.array([float(hdu[0].header['CDELT1']), float(hdu[0].header['CDELT2'])])
    w.wcs.crval = [float(hdu[0].header['CRVAL1']), float(hdu[0].header['CRVAL2'])]
    w.wcs.ctype = [hdu[0].header['CTYPE1'], hdu[0].header['CTYPE2']]
    w.wcs.cunit = ['deg', 'deg']
    w.wcs.pc = [[-1.0, 0.0], [ 0.0, 1.0]]
    wcs_header = w.to_header()
    hdu.close()

    # get cube photometry
    cube_photo = np.zeros((nbands,dimy,dimx))
    cube_photo_err = np.zeros((nbands,dimy,dimx))

    for bb in range(nbands):
        hdu1 = fits.open(output_stamps['name_img_'+filters[bb]])
        hdu2 = fits.open(output_stamps['name_var_'+filters[bb]])

        # convert flux unit into erg/s/cm^2/Angstrom
        if image_unit == 'erg/s/cm2/A':
            cube_photo[bb,:,:] = hdu1[0].data*image_scale[filters[bb]]
            cube_photo_err[bb,:,:] = np.sqrt(hdu2[0].data)*image_scale[filters[bb]]
        elif image_unit == 'Jy':
            cube_photo[bb,:,:] = hdu1[0].data*1.0e-23*2.998e+18*image_scale[filters[bb]]/photo_wave[bb]/photo_wave[bb]
            cube_photo_err[bb,:,:] = np.sqrt(hdu2[0].data)*1.0e-23*2.998e+18*image_scale[filters[bb]]/photo_wave[bb]/photo_wave[bb]
        elif image_unit == 'MJy/sr':
            f0 = hdu1[0].data*2.350443e-5*img_pixsizes[add_fil]*img_pixsizes[add_fil]    # in unit of Jy
            cube_photo[bb,:,:] = f0*1.0e-23*2.998e+18*image_scale[filters[bb]]/photo_wave[bb]/photo_wave[bb]   # in erg/s/cm^2/Ang.

            f0 = np.sqrt(hdu2[0].data)*2.350443e-5*img_pixsizes[add_fil]*img_pixsizes[add_fil]    # in unit of Jy
            cube_photo_err[bb,:,:] = f0*1.0e-23*2.998e+18*image_scale[filters[bb]]/photo_wave[bb]/photo_wave[bb]   # in erg/s/cm^2/Ang.

        else:
            print ('image_unit is not recognized!')
            sys.exit()

        hdu1.close()
        hdu2.close()

    ####===> PSF matching for the NIRSpec IFU data cube <===####
    # structure is: [ch,wave,y,x]
    conv_cube_header = {}
    conv_cube_wave = {}
    conv_cube_spec = {}
    conv_cube_err = {}

    for ch in disperser_filter:
        if verbose==True and flag_psfmatch==0:
            print ('perform PSF matching for the NIRSpec IFU %s data...' % ch)

        cube = fits.open(nirspec_ifu[ch])
        conv_cube_header[ch] = cube['sci'].header
        
        wave_um = get_wavelength_grid_nirspec(cube['sci'].header)
        wave_a = wave_um*1e+4
        conv_cube_wave[ch] = wave_um

        conv_cube_spec[ch] = np.zeros(cube['sci'].data.shape)
        conv_cube_err[ch] = np.zeros(cube['sci'].data.shape)

        # open PSF cube for this channel
        cube1 = fits.open(nirspec_psf_cubes[ch])
        cube_psf = cube1['DET_DIST'].data
        cube_psf_wave = np.zeros(cube_psf.shape[0])
        for i in range(cube_psf.shape[0]):
            cube_psf_wave[i] = cube1['det_dist'].header["WVLN%04d" % i]*1e+6
        cube1.close()

        for ww in range(len(wave_um)):
            if flag_psfmatch == 1:
                # No PSF matching
                if nirspec_unit == 'erg/s/cm2/A':
                    conv_cube_spec[ch][ww,:,:] = cube['sci'].data[ww,:,:]*nirspec_scale[ch]
                    conv_cube_err[ch][ww,:,:] = cube['err'].data[ww,:,:]*nirspec_scale[ch]

                elif nirspec_unit == 'Jy':
                    conv_cube_spec[ch][ww,:,:] = cube['sci'].data[ww,:,:]*nirspec_scale[ch]*1.0e-23*2.998e+18/wave_a[ww]/wave_a[ww]    # in erg/s/cm^2/Ang.
                    conv_cube_err[ch][ww,:,:] = cube['err'].data[ww,:,:]*nirspec_scale[ch]*1.0e-23*2.998e+18/wave_a[ww]/wave_a[ww]     # in erg/s/cm^2/Ang.
   
                elif nirspec_unit == 'MJy/sr':
                    f0 = cube['sci'].data[ww,:,:]*2.350443e-5*img_pixsizes[add_fil]*img_pixsizes[add_fil]    # in unit of Jy
                    conv_cube_spec[ch][ww,:,:] = f0*1.0e-23*2.998e+18*nirspec_scale[ch]/wave_a[ww]/wave_a[ww]   # in erg/s/cm^2/Ang.

                    f0 = cube['err'].data[ww,:,:]*2.350443e-5*img_pixsizes[add_fil]*img_pixsizes[add_fil]    # in unit of Jy
                    conv_cube_err[ch][ww,:,:] = f0*1.0e-23*2.998e+18*nirspec_scale[ch]/wave_a[ww]/wave_a[ww]   # in erg/s/cm^2/Ang.
            else:
                if ch==fil_max_psf and ww==len(wave_um)-1:
                    # No PSF matching
                    if nirspec_unit == 'erg/s/cm2/A':
                        conv_cube_spec[ch][ww,:,:] = cube['sci'].data[ww,:,:]*nirspec_scale[ch]
                        conv_cube_err[ch][ww,:,:] = cube['err'].data[ww,:,:]*nirspec_scale[ch]

                    elif nirspec_unit == 'Jy':
                        conv_cube_spec[ch][ww,:,:] = cube['sci'].data[ww,:,:]*nirspec_scale[ch]*1.0e-23*2.998e+18/wave_a[ww]/wave_a[ww]    # in erg/s/cm^2/Ang.
                        conv_cube_err[ch][ww,:,:] = cube['err'].data[ww,:,:]*nirspec_scale[ch]*1.0e-23*2.998e+18/wave_a[ww]/wave_a[ww]     # in erg/s/cm^2/Ang.
    
                    elif nirspec_unit == 'MJy/sr':
                        f0 = cube['sci'].data[ww,:,:]*2.350443e-5*img_pixsizes[add_fil]*img_pixsizes[add_fil]    # in unit of Jy
                        conv_cube_spec[ch][ww,:,:] = f0*1.0e-23*2.998e+18*nirspec_scale[ch]/wave_a[ww]/wave_a[ww]   # in erg/s/cm^2/Ang.

                        f0 = cube['err'].data[ww,:,:]*2.350443e-5*img_pixsizes[add_fil]*img_pixsizes[add_fil]    # in unit of Jy
                        conv_cube_err[ch][ww,:,:] = f0*1.0e-23*2.998e+18*nirspec_scale[ch]/wave_a[ww]/wave_a[ww]   # in erg/s/cm^2/Ang.

                else:
                    # get the PSF for this wavelength
                    if wave_um[ww] < max(cube_psf_wave):
                        psf_interp = interpolate_psf_cube(cube_psf, cube_psf_wave, wave_um[ww], kind='cubic')
                    else:
                        psf_interp = cube_psf[-1,:,:]
                    
                    # calculate the kernel for PSF matching
                    kernel = create_psf_matching_kernel(psf_interp, max_psf, img_pixsizes[add_fil], max_psf_pixsize, 
                                                        window=kernel_window, window_arg=kernel_window_arg)

                    # convolve the slice image with the PSF kernel
                    conv_sci = convolve_fft(cube['sci'].data[ww,:,:], kernel, allow_huge=True)
                    conv_err = convolve_fft(cube['err'].data[ww,:,:], kernel, allow_huge=True)

                    if nirspec_unit == 'erg/s/cm2/A':
                        conv_cube_spec[ch][ww,:,:] = conv_sci*nirspec_scale[ch]
                        conv_cube_err[ch][ww,:,:] = conv_err*nirspec_scale[ch]
                    elif nirspec_unit == 'Jy': 
                        conv_cube_spec[ch][ww,:,:] = conv_sci*nirspec_scale[ch]*1.0e-23*2.998e+18/wave_a[ww]/wave_a[ww]
                        conv_cube_err[ch][ww,:,:] = conv_err*nirspec_scale[ch]*1.0e-23*2.998e+18/wave_a[ww]/wave_a[ww]
                    elif nirspec_unit == 'MJy/sr':
                        f0 = conv_sci*2.350443e-5*img_pixsizes[add_fil]*img_pixsizes[add_fil]
                        conv_cube_spec[ch][ww,:,:] = f0*1.0e-23*2.998e+18*nirspec_scale[ch]/wave_a[ww]/wave_a[ww]

                        f0 = conv_err*2.350443e-5*img_pixsizes[add_fil]*img_pixsizes[add_fil]
                        conv_cube_err[ch][ww,:,:] = f0*1.0e-23*2.998e+18*nirspec_scale[ch]/wave_a[ww]/wave_a[ww]

            if verbose==True and flag_psfmatch==0:
                sys.stdout.write('\r')
                sys.stdout.write('wavelength grid: %d of %d (%d%%)' % (ww+1,len(wave_um),(ww+1)*100/len(wave_um)))
                sys.stdout.flush()
        if verbose==True and flag_psfmatch==0:
            sys.stdout.write('\n')                     

        cube.close()

    ## merge ifu cubes across channels if there are multiple of them, and resample the wavelength
    if n_ifu > 1:
        if verbose:
            print ('merging NIRSpec IFU cubes across channels and resample wavelength...')

        merge_cube_wave_a = np.arange(min(conv_cube_wave[disperser_filter[0]])*1e+4, max(conv_cube_wave[disperser_filter[n_ifu-1]])*1e+4, del_wave_resamp)
        merge_cube_wave_um = merge_cube_wave_a/1e+4

        merge_cube_spec = np.zeros((len(merge_cube_wave_um),dimy,dimx))
        merge_cube_err = np.zeros((len(merge_cube_wave_um),dimy,dimx))

        num_div = np.zeros(len(merge_cube_wave_um))
        for mm in range(n_ifu):
            ids1 = np.where((merge_cube_wave_um>=min(conv_cube_wave[disperser_filter[mm]])) & (merge_cube_wave_um<=max(conv_cube_wave[disperser_filter[mm]])))[0]
            num_div[ids1] += 1

        for yy in range(dimy):
            for xx in range(dimx):
                tot_spec = np.zeros(len(merge_cube_wave_um))
                tot_err2 = np.zeros(len(merge_cube_wave_um))
                for mm in range(n_ifu):
                    temp_spec, temp_err = spectres(merge_cube_wave_a, conv_cube_wave[disperser_filter[mm]]*1e+4, conv_cube_spec[disperser_filter[mm]][:,yy,xx], 
                                                   spec_errs=conv_cube_err[disperser_filter[mm]][:,yy,xx], fill=0, verbose=True)
                    tot_spec += temp_spec
                    tot_err2 += np.square(temp_err)

                merge_cube_spec[:,yy,xx] = tot_spec/num_div 
                merge_cube_err[:,yy,xx] = np.sqrt(tot_err2)/num_div

    ##=> perform spectral matching between photometric SED and spectral SED <===
    if verbose:
        print ('perform spectral matching between photometric SED and spectral SED...')
    
    # get photometric SEDs of pixels 
    pix_rows, pix_cols = np.where(nirspec_region == 1)
    #xc, yc = (dimx-1)/2, (dimy-1)/2
    #pix_rows = []
    #pix_cols = []
    #for yy1 in range(int(yc)-3, int(yc)+4):
    #    for xx1 in range(int(xc)-3, int(xc)+4):
    #        pix_rows.append(yy1)
    #        pix_cols.append(xx1)
    n_pix = len(pix_rows)
    if verbose:
        print ('number of pixels with photometry and spectra: %d' % n_pix)

    photo_seds = np.zeros((n_pix, nbands))
    photo_seds_err = np.zeros((n_pix, nbands))
    for bb in range(nbands):
        photo_seds[:,bb] = cube_photo[bb,pix_rows,pix_cols]
        photo_seds_err[:,bb] = cube_photo_err[bb,pix_rows,pix_cols]

    # get spectral SEDs of pixels
    if n_ifu > 1:
        spec_wave_angstrom = merge_cube_wave_a
    else:
        spec_wave_angstrom = conv_cube_wave[disperser_filter[0]]*1e+4

    spec_seds = np.zeros((n_pix, len(spec_wave_angstrom)))
    spec_seds_err = np.zeros((n_pix, len(spec_wave_angstrom)))

    for ii in range(n_pix):
        if n_ifu > 1:
            spec_seds[ii,:] = merge_cube_spec[:,pix_rows[ii],pix_cols[ii]]
            spec_seds_err[ii,:] = merge_cube_err[:,pix_rows[ii],pix_cols[ii]]
        else:
            spec_seds[ii,:] = conv_cube_spec[disperser_filter[0]][:,pix_rows[ii],pix_cols[ii]]  
            spec_seds_err[ii,:] = conv_cube_err[disperser_filter[0]][:,pix_rows[ii],pix_cols[ii]]

    # determine spectral resolution
    spec_resolution = 3000
    for ch in disperser_filter:
        sr = get_spec_resolution_disperser_filter(ch)
        if sr < spec_resolution:
            spec_resolution = sr

    if verbose:
        print ('spectral resolution is set to %d' % spec_resolution)

    output1 = spectral_matching_specphoto(gal_z, filters, photo_seds, photo_seds_err, spec_wave_angstrom, spec_seds, spec_seds_err, ncpu=ncpu, 
                                          nmodels=100000, spec_resolution=spec_resolution, del_wave_nebem=del_wave_nebem, poly_order=poly_order, 
                                          verbose=verbose)
    corr_spec_seds, corr_factor = output1

    # correct spectra of merged NIRSpec IFU cubes
    if n_ifu > 1:
        merge_cube_spec_corr = np.zeros(merge_cube_spec.shape)
        for ii in range(n_pix):
            merge_cube_spec_corr[:,pix_rows[ii],pix_cols[ii]] = corr_spec_seds[ii]

    # correct spectra of individual NIRSpec IFU cubes
    conv_cube_spec_corr = {}
    for ch in disperser_filter:
        conv_cube_spec_corr[ch] = np.zeros(conv_cube_spec[ch].shape)
        for ii in range(n_pix):
            func = interp1d(spec_wave_angstrom, corr_factor[ii], fill_value='extrapolate', bounds_error=False)
            interp_corr_factor = func(conv_cube_wave[ch]*1e+4)

            conv_cube_spec_corr[ch][:,pix_rows[ii],pix_cols[ii]] = interp_corr_factor + conv_cube_spec[ch][:,pix_rows[ii],pix_cols[ii]]     ####

    # store correction factors
    cube_corr_factor = np.zeros((corr_factor.shape[1],dimy,dimx))
    for ii in range(n_pix):
        cube_corr_factor[:,pix_rows[ii],pix_cols[ii]] = corr_factor[ii]

    wcs_header['nfilters'] = nbands
    wcs_header['z'] = gal_z
    wcs_header['RA'] = gal_ra 
    wcs_header['DEC'] = gal_dec 
    wcs_header['unit'] = 1.0
    wcs_header['bunit'] = 'erg/s/cm^2/A'
    wcs_header['structph'] = '(band,y,x)'
    wcs_header['structsp'] = '(wavelength,y,x)'
    wcs_header['pixsize'] = img_pixsizes[add_fil]
    wcs_header['specphot'] = 1
    wcs_header['ifssurve'] = 'nirspec_ifu'
    for ii in range(nbands):
        wcs_header['fil'+str(ii)] = filters[ii]
    for ii in range(n_ifu):
        wcs_header['sfil'+str(ii)] = disperser_filter[ii]

    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU(header=wcs_header))

    # photometric data cube
    hdul.append(fits.ImageHDU(data=cube_photo, name='photo_flux'))
    hdul.append(fits.ImageHDU(data=cube_photo_err, name='photo_fluxerr'))

    # NIRSpec region
    hdul.append(fits.ImageHDU(data=nirspec_region, name='spec_region'))

    # individual NIRSpec IFU cubes without correction
    for ch in disperser_filter:
        hdul.append(fits.ImageHDU(data=conv_cube_wave[ch], name=ch+'_wave'))
        hdul.append(fits.ImageHDU(data=conv_cube_spec[ch], header=conv_cube_header[ch], name=ch+'_flux'))
        hdul.append(fits.ImageHDU(data=conv_cube_err[ch], header=conv_cube_header[ch], name=ch+'_fluxerr'))
    
    # merged NIRSpec IFU cube with correction
    if n_ifu > 1:
        hdul.append(fits.ImageHDU(data=merge_cube_wave_um, name='spec_wave'))
        hdul.append(fits.ImageHDU(data=merge_cube_spec, name='spec_flux'))
        hdul.append(fits.ImageHDU(data=merge_cube_err, name='spec_fluxerr'))

    # individual NIRSpec IFU cubes with correction
    for ch in disperser_filter:
        hdul.append(fits.ImageHDU(data=conv_cube_wave[ch], name=ch+'_wave_corr'))
        hdul.append(fits.ImageHDU(data=conv_cube_spec_corr[ch], name=ch+'_flux_corr'))  

    # merged NIRSpec IFU cube with correction
    if n_ifu > 1:
        hdul.append(fits.ImageHDU(data=merge_cube_wave_um, name='spec_wave_corr'))
        hdul.append(fits.ImageHDU(data=merge_cube_spec_corr, name='spec_flux_corr'))

    # correction factor
    hdul.append(fits.ImageHDU(data=cube_corr_factor, name='corr_factor')) 

    if name_out_fits is None:
        name_out_fits = 'combined_datacube.fits'
    
    if verbose:
        print ('Writing output datacube to %s...' % name_out_fits)
    
    hdul.writeto(name_out_fits, overwrite=True)




