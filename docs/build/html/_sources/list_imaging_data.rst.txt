List of imaging data
=====================

The list of imaging dataset that can be analyzed by the current version of **piXedfit** 
and a brief description on their specifications, unit of pixel value, and how to estimate flux uncertainty are given in the following. 

*   **Galaxy Evolution Explorer (GALEX)**
    
    The input image (in FITS) is assumed to have the same format as the one obtained from the `MAST <http://galex.stsci.edu/GR6/>`_. 
    Commonly, the background subtraction has been done for the imaging data product and the background image is provided in a seperate FITS file.     
    The imaging data in the two bands (FUV and NUV) have spatial resolution (i.e., FWHM of PSF) of 4.2'' and 5.3'', respectively. 
    The spatial sampling is 1.5''/pixel. The :math:`5\sigma` limiting magnitudes in FUV (NUV) of the three surveys modes (AIS, MIS, and DIS)
    are 19.9 (20.8), 22.6 (22.7), 24.8 (24.4), respectively. Pixel value of the imaging data is in unit of counts (i.e., number of detected photons) per second (CPS).
    For more information, please refer to `Morrissey et al. (2007) <https://ui.adsabs.harvard.edu/abs/2007ApJS..173..682M/abstract>`_.
    To convert the pixel value into flux and estimate flux uncertainty,
    we follow the relevant information from the literature and the survey's website.
    To convert from pixel value to flux in unit of :math:`\text{erg }\text{s}^{-1}\text{cm}^{-2}{\text{Å}}^{-1}`, the following equation is used:

        *   **FUV**:    :math:`\text{flux} = 1.40 \times 10^{-15} \text{CPS}`
        *   **NUV**:    :math:`\text{flux} = 2.06 \times 10^{-16} \text{CPS}`

    
    To get flux uncertainty, first, uncertainty of counts is estimated using the following equation:

        *   **FUV**:    :math:`\text{CPS}_{\text{err}} = \frac{\sqrt{\text{CPS}\times \text{exp-time} + (0.050\times \text{CPS}\times \text{exp-time})^{2}}}{\text{exp-time}}`
        *   **NUV**:    :math:`\text{CPS}_{\text{err}} = \frac{\sqrt{\text{CPS}\times \text{exp-time} + (0.027\times \text{CPS}\times \text{exp-time})^{2}}}{\text{exp-time}}`         

    The exp-time is exposure time which can be obained from the FITS header (keyword: EXPTIME). Then flux uncertainty can be calculated using the above equations for 
    converting from counts to flux. The above information is taken from GALEX's `website <https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html>`_.


*   **Sloan Digital Sky Survey (SDSS)** 

    The input image (in FITS) is assumed to have the same format as that of the `Corrected Frame <https://dr16.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html>`_ product of SDSS. 
    For this imaging data product, background subtraction has been done and the background image can be reconstructed from a cruder 2D grid of background image sstored in the HDU2 extension of the FITS file.
    The spatial sampling of the imaging data in the 5 bands (:math:`u`, :math:`g`, :math:`r`, :math:`i`, and :math:`z`) is 0.396''/pixel. 
    The median seeing of all SDSS imaging data is 1.32'' in the :math:`r`-band (see `Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_).
    The SDSS imaging is 95% complete to :math:`u=22.0` mag, :math:`g=22.2` mag, :math:`r=22.2` mag, :math:`i=21.3` mag, and :math:`z=20.5` mag (`Abazajian et al. 2004 <https://ui.adsabs.harvard.edu/abs/2004AJ....128..502A/abstract>`_).
    The pixel value in the SDSS image is counts in unit of nanomaggy. To convert the pixel value into flux in unit of :math:`\text{erg }\text{s}^{-1}\text{cm}^{-2}{\text{Å}}^{-1}`, 
    the following equation is used:
    
            :math:`\text{flux} = \text{counts}\times 3.631\times 10^{6}\times 2.994\times 10^{-5} \left(\frac{1}{\lambda_{c}}\right)^{2}` 
    
    where the :math:`\lambda_{c}` is the central wavelength of the photometric band.

    To estimate flux uncertainty of a pixel, first, the uncertainty of counts is calculated using the following equation:

            :math:`\text{counts}_{\text{err}} = \sqrt{\frac{\left(\frac{counts}{NMGY}\right) + \text{counts}_{\text{sky}}}{\text{gain}} + \text{dark variance}}`   

    with NMGY is a conversion factor from counts to flux in unit of nanomaggy (i.e., nanomaggy per count) and :math:`\text{count}_{\text{sky}}` is counts
    associated with the sky background image. The :math:`\text{count}_{\text{sky}}` at a particular coordinate in the image is obtained from bilinear interpolation 
    to the cruder 2D grids of sky background counts stored in the HDU2 of the FITS file. Gain is a convertion factor from count to the detected number of 
    photo electron and dark variance is an additional source of noise from the read-noise and the noise in the dark current. Values of gain and dark variance 
    vary depending on the camera column (camcol) and the photometric band. Those values can be obtained from this SDSS's `web page <https://dr16.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html>`_.
    After getting the :math:`\text{count}_{\text{err}}`, the flux uncertainty can be calculated using the following equation:

            :math:`\text{flux_err} = \text{counts}_{\text{err}} \times 3.631\times 10^{6}\times 2.994\times 10^{-5} \left(\frac{1}{\lambda_{c}}\right)^{2}`   

    The above information is obtained from this SDSS's `web page <https://dr16.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html>`_. 


*   **Hubble Space telescope (HST)**

    The HST image has a spatial sampling of 0.06''/pixel. The PSF FWHM varies across photometric bands. The PSF FWHM of F160W band is 0.19''. 
    The :math:`5\sigma` limiting magnitude of F160W is 26.4 mag. Pixel value of the HST image is counts per second. To convert the pixel value 
    to flux in unit of :math:`\text{erg }\text{s}^{-1}\text{cm}^{-2}{\text{Å}}^{-1}`, a multiplicative convertion factor can be found in the 
    header of the FIST file (keyword: PHOTFLAM). Flux uncertainty of a pixel can be calculated from the weight image, which commonly provided 
    by various surveys.   


*   **Two Micron All Sky Survey (2MASS)**

    The input image (in FITS) is assumed to be in the same format as that provided by the NASA/IPAC Infrared Science Archive (`IRSA <https://irsa.ipac.caltech.edu/applications/2MASS/IM/>`_).
    Commonly, the imaging data from that source is not background-subtracted.
    The imaging data product has spatial sampling of 1.0''/pixel. The seeing is :math:`\sim 2.5-3.5`'' (`Skrutskie et al. 2006 <https://ui.adsabs.harvard.edu/abs/2006AJ....131.1163S/abstract>`_).
    The point-source sensitivities at signal-to-noise ratio S/N=10 are: 15.8, 15.1, and 14.3 mag for :math:`J`, :math:`H`, and :math:`\text{K}_{\text{S}}`, respectively.
    Pixel value of the 2MASS image is in data-number unit (DN). To convert the pixel value to magnitude, one need a magnitude zero-point which can be obtained 
    from the header of the FITS file (keyword: MAGZP). Then the flux can be calculated using a flux for zero-magnitude zero-point conversion values (:math:`f_{\lambda,\text{zero-mag}}`). 
    The :math:`f_{\lambda,\text{zero-mag}}` in unit of :math:`\text{W }\text{cm}^{-2}\mu\text{m}^{-1}` for the :math:`J`, :math:`H`, and :math:`\text{K}_{\text{S}}` bands 
    are :math:`3.129\times 10^{-13} \pm 5.464\times 10^{-15}`, :math:`1.133\times 10^{-13}\pm 2.212\times 10^{-15}`, and :math:`4.283\times 10^{-14}\pm 8.053\times 10^{-16}`, respectively (see this `web page <https://old.ipac.caltech.edu/2mass/releases/allsky/doc/sec6_4a.html>`_).
    A proper conversion factor is then applied to convert the flux in the :math:`\text{W }\text{cm}^{-2}\mu\text{m}^{-1}` to :math:`\text{erg }\text{s}^{-1}\text{cm}^{-2}{\text{Å}}^{-1}`. 
    The flux calibration of 2MASS is described in `Cohen et al. (2003) <https://ui.adsabs.harvard.edu/abs/2003AJ....126.1090C/abstract>`_. The uncertainty of pixel value in a 2MASS 
    image is estimated following the procedure described in the 2MASS survey's website `here <https://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/>`_ (see the handy equations in the botttom of the web page). 


*   **Wide-field Infrared Survey Explorer (WISE)**

    The input image (in FITS) is assumed to be in the same format as that provided in this IRSA `website <https://irsa.ipac.caltech.edu/applications/wise/>`_.
    Commonly, the imaging data from that source is not background-subtracted.
    The imaging data in the 4 bands :math:`3.4\mu \text{m}` (:math:`W1`), :math:`4.6\mu \text{m}` (:math:`W2`), :math:`12\mu \text{m}` (:math:`W3`), and :math:`22\mu \text{m}` (:math:`W4`) have 
    spatial resolutions of 6.1'', 6.4'', 6.5'', and 12.0'', respectively. The spatial sampling of the imaging data in the 4 bands is 1.375''/pixel.
    WISE achieved :math:`5\sigma` point source sensitivities better than 0.08, 0.11, 1.00, and 6.00 mJy in unconfused regions on the ecliptic in the 4 bands (`Wright et al. 2010 <https://ui.adsabs.harvard.edu/abs/2010AJ....140.1868W/abstract>`_).
    The pixel value of a WISE image is DN unit. To convert the pixel value to flux in Jy, one need *DN_to_Jy* conversion factor. The *DN_to_Jy* for the 
    :math:`W1`, :math:`W2`, :math:`W3`, and :math:`W4` are :math:`1.935\times 10^{-6}`, :math:`2.7048\times 10^{-6}`, :math:`1.8326\times 10^{-6}`, and 
    :math:`5.2269\times 10^{-5}`, respectively. The flux is then converted from Jy to :math:`\text{erg }\text{s}^{-1}\text{cm}^{-2}{\text{Å}}^{-1}`.
    The WISE image atlas product commonly provides uncertainty image that gives the propagated :math:`1\sigma` uncertainty estimate for each pixel in the 
    corresponding coadded intensity image. For estimating flux uncertainty, a relevant instruction is given in a WISE survey's website `here <https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html>`_.   


*   **Spitzer (IRAC and MIPS)**

    The mean FWHMs of the PSFs of the 4 IRAC bands :math:`3.6\mu \text{m}`, :math:`4.5\mu \text{m}`, :math:`5.8\mu \text{m}`, and :math:`8.0\mu \text{m}` are 
    1.66'', 1.72'', 1.88'', and 1.98'', respectively (`Fazio et al. 2004 <https://ui.adsabs.harvard.edu/abs/2004ApJS..154...10F/abstract>`_). The mean FWHMs of the 
    PSFs of the 3 MIPS bands :math:`24\mu \text{m}`, :math:`70\mu \text{m}`, and :math:`160\mu \text{m}` are 6.0'', 18.0'', and 40'', respectively (`Rieke et al. 2004 <https://ui.adsabs.harvard.edu/abs/2004ApJS..154...25R/abstract>`_). 
    The spatial sampling of IRAC imaging data is 1.2''/pixel, while the spatial sammpling of MIPS imaging data varies across bands: 1.5''/pixel (:math:`24\mu \text{m}`), 
    4.5''/pixel (:math:`70\mu \text{m}`), and 9.0''/pixel (:math:`160\mu \text{m}`). The :math:`1\sigma` point-source sensitivities (with low background and 100 second time frame) of the 4 IRAC bands are 
    :math:`0.6\mu \text{Jy}` (:math:`3.6\mu \text{m}`), :math:`1.2\mu \text{Jy}` (:math:`4.5\mu \text{m}`), :math:`8.0\mu \text{Jy}` (:math:`5.8\mu \text{m}`), and 
    :math:`9.8\mu \text{Jy}` (:math:`8.0\mu \text{m}`) (`Fazio et al. 2004 <https://ui.adsabs.harvard.edu/abs/2004ApJS..154...10F/abstract>`_).
    The pre-launch estimate of the :math:`1\sigma` confusion limits of the MIPS bands are :math:`\sim 0.5-1.3 \text{mJy}` (:math:`70\mu \text{m}`) and 
    :math:`\sim 7.0-19.0 \text{mJy}` (:math:`160\mu \text{m}`) (`Xu et al. 2001 <https://ui.adsabs.harvard.edu/abs/2001ApJ...562..179X/abstract>`_ and `Dole et al. 2003 <https://iopscience.iop.org/article/10.1086/346130>`_).
    Pixel values of the IRAC and MIPS are in unit of Mjy/sr. To convert the pixel value to flux density in :math:`\text{erg }\text{s}^{-1}\text{cm}^{-2}{\text{Å}}^{-1}`, one 
    needs pixel size of the image. For estimating the flux uncertainty of a pixel, we use the uncertainty map (commonly provided by surveys, such as SINGS; `Kennicutt et al. 2003 <https://ui.adsabs.harvard.edu/abs/2003PASP..115..928K/abstract>`_) 
    whenever available. If the uncertainty map is not available, the flux uncertainty is assumed to be dominated by the calibration uncertainty. 
    The calibration uncertainty of the 4 bands of IRAC is :math:`\sim` 10% (`Reach et al. 2005 <https://ui.adsabs.harvard.edu/abs/2005PASP..117..978R/abstract>`_; `Munoz-Mateos <https://ui.adsabs.harvard.edu/abs/2009ApJ...703.1569M/abstract>`_), 
    whereas that uncertainties for the 3 bands of MIPS are 4% (:math:`24\mu \text{m}`), 5% (:math:`70\mu \text{m}`), and 12% (:math:`160\mu \text{m}`) 
    (`Engelbracht et al. 2007 <https://ui.adsabs.harvard.edu/abs/2007PASP..119..994E/abstract>`_; `Gordon et al. 2007 <https://ui.adsabs.harvard.edu/abs/2007PASP..119.1019G/abstract>`_; `Stansberry et al. 2007 <https://ui.adsabs.harvard.edu/abs/2007PASP..119.1038S/abstract>`_).


*   **Herschel (PACS and SPIRE)**  

    The three PACS bands have measured PSF FWHMs of 5.67'' (:math:`70\mu \text{m}`), 7.04'' (:math:`100\mu \text{m}`), and 11.18'' (:math:`160\mu \text{m}`) 
    (`Anianp et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_, Geis N. and Lutz D. 2010 PACS ICC Document PICC-ME-TN-029 v2.0, Lutz D. 2010 PACS ICC Document PICC-ME-TN-033, and Müller T. 2010 PACS ICC Document PICC-ME-TN-036 v2.0).  
    The three SPIRE bands have mean PSF FWHMs of 18.1'' (:math:`250\mu \text{m}`), 25.2'' (:math:`350\mu \text{m}`), and 36.6'' (:math:`500\mu \text{m}`).
    The measured confusion noise levels in the :math:`250\mu \text{m}`, :math:`350\mu \text{m}`, and :math:`500\mu \text{m}` bands are 5.8 mJy, 6.3 mJy, and 6.8 mJy, respectively 
    (`Griffin ett al. 2010 <https://ui.adsabs.harvard.edu/abs/2010A%26A...518L...3G/abstract>`_). The PACS imaging data has pixel value in the unit of 
    Jy/pixel, while the SPIRE imaging data varies depending on the survey from which the data is obtained. The SPIRE imaging data provided by the 
    Key Insights on Nearby Galaxies: A Far-Infrared Survey with Herschel (KINGFISH; `Kennicutt et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1347K/abstract>`_) 
    has pixel value in the unit of MJy/sr, whereas the SPIRE imaging data provided by the Very Nearby Galaxy Survey (VNGS; `Bendo et al. 2012 <https://academic.oup.com/mnras/article/419/3/1833/1061333>`_) 
    has pixel value in the unit of Jy/beam. Based on the SPIRE Observer's Manual, the beam areas in :math:`\text{arcsec}^{2}` of the :math:`250\mu \text{m}`, :math:`350\mu \text{m}`, and 
    :math:`500\mu \text{m}` are 426, 771, and 1626, respectively. For estimating the flux uncertainty of a pixel, an uncertainty map (such as that provided 
    by the KINGFISH and VNGS surveys) is used whenever available. Otherwise, the flux uncertainty is estimated by assuming that the flux uncertainty is 
    dominated by the calibration uncertainty. The calibration uncertainty of PACS is :math:`\sim` 5% (according to the version 4 of the `PACS Observer's Manual <http://herschel.esac.esa.int/Docs/PACS/html/pacs_om.html>`_), while 
    the calibration uncertainty of the SPIRE is :math:`\sim` 7% (see this `web page <https://www.cosmos.esa.int/web/herschel/legacy-documentation-spire/>`_).    


    
