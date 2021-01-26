piXedfit
========
**piXedfit** is a Python package that provides a self-contained set of tools for analyzing spatially resolved properties of galaxies using 
imaging data or a combination of imaging data and the integral field spectroscopy (IFS) data. **piXedfit** has six modules that can 
handle all tasks in the analysis of spatially resolved SEDs of a galaxy, including images processing, a spatial-matching (in spatial resolution and sampling) between broad-band images and IFS data cube, pixel binning, performing SED fitting, and making visualization plots for the SED fitting results. 
**piXedfit** is a versatile tool that has been equipped with the multiprocessing module (namely message passing interface or MPI) for 
efficient analysis of the datasets of a large number of galaxies. Detailed description of **piXedfit** and its performances are presented in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.  

While this website is still under construction, people interested in knowing how **piXedfit** works can see some demonstrations from a folder examples on the `GitHub page <https://github.com/aabdurrouf/piXedfit>`_. Some animations can also be seen from: `images processing <https://github.com/aabdurrouf/piXedfit/blob/main/docs/source/demos_img_pros.rst>`_, `pixel binning <https://github.com/aabdurrouf/piXedfit/blob/main/docs/source/demos_pixel_binning.rst>`_ and `SED fitting <https://github.com/aabdurrouf/piXedfit/blob/main/docs/source/demos_sed_fitting.rst>`_.   

.. image:: 3Dcube_specphoto.png
.. image:: demo_pixedfit_ngc309_new_edit.svg
.. image:: plot_maps_props_new.svg
   :width: 600

Features
--------
**piXedfit** has 6 modules that can work independently with each other such that a user interested in using a particular module in **piXedfit** 
doesn't need to use the other modules. For instance, it is possible to use the SED fitting module for fitting observed SED, either integrated 
of spaially resolved SED, without the need of using the images processing and pixel binning modules. The 6 modules and their usabilities are the following:

*  **piXedfit_images: image processing**
   
   This module is capable of doing spatial-matching (in spatial resolution and spatial sampling) of multiband images ranging from the FUV to FIR 
   (from ground-based and spaced-based telescopes) and extract pixel-wise photometric SEDs within the galaxy's region of interest.

*  **piXedfit_spectrophotometric: spatial-matching of imaging data and the IFS data**

   This module is capable of doing spatial-matching (in spatial resolution and sampling) of a multiband imaging data 
   (that have been processed by the :mod:`piXedfit_images`) with an IFS data cube (containing the same galaxy) and extract pixel-wise 
   spectrophotometric SEDs within the galaxy's region of interest. For the current version of **piXedfit**, only the IFS data from 
   the `CALIFA <https://califa.caha.es/>`_ and `MaNGA <https://www.sdss.org/surveys/manga/>`_ can be analyzed with the 
   :mod:`piXedfit_spectrophotometric` module.   

*  **piXedfit_bin: pixel binning**

   This module is capable of performing pixel binning, which is a process of combining neighboring pixels to achieve certain S/N thresholds.
   The pixel binning scheme takes into account the similarity of SED shape among the pixels that are going to be binned together. This way 
   important spatial information from the pixel scale can be expected to be preserved. 
   The S/N threshold can be set to all the bands, not limited to a particular band.   

*  **piXedfit_model: generating model SEDs**

   This module can generate model SEDs of galaxies given some parameters. The SED modeling uses the `FSPS <https://github.com/cconroy20/fsps>`_ SPS model 
   with the `Python-FSPS <http://dfm.io/python-fsps/current/>`_ as the interface to the Python environment. The SED modeling incorporates the modeling 
   of light coming from stellar emission, nebular emission, dust emission, and the AGN dusty torus emission.      

*  **piXedfit_fitting: performing SED fitting**

   This module is capable of performing SED fitting for input SEDs of galaxies, either spatially resolved SED or integrated SED. The input can be 
   in the form of photometric SED or spetrophotometric SED (i.e., combination of photometry and spectroscopy). When fed with a spectrophotometric SED, 
   **piXedfit** can simultaneously fit the photometric and spectroscopic SEDs. 

*  **piXedfit_analysis: making visualization plots for the SED fiting results**

   This module can make three plots for visualizing the fitting results: corner plot (i.e., a plot showing 1D and joint 2D posteriors of the parameters), SED plot (i.e., a plot showing recovery of the input SED by the best-fit model SED), and SFH plot (i.e., a plot showing inferred SFH from the fitting).


.. toctree::
   :maxdepth: 2
   :caption: User guide

   install
   list_imaging_data
   list_kernels_psf
   image_pros
   image_ifs_match
   pixel_binning
   ingredients_model
   gen_model_SEDs
   fit_res_SEDs
   fit_ing_SEDs
   plot_fitres
   get_maps_prop


.. toctree::
   :maxdepth: 2
   :caption: Demonstrations and tutorials

   demos_img_pros
   demos_pixel_binning
   demos_sed_fitting
   tutorials


.. toctree::
   :maxdepth: 2
   :caption: API reference 

   piXedfit_images
   piXedfit_spectrophotometric
   piXedfit_bin
   piXedfit_model
   piXedfit_fitting
   piXedfit_analysis



Reference
---------

A list of some projects **piXedfit** is benefitted from:

*  `FSPS <https://github.com/cconroy20/fsps>`_ and `Python-FSPS <http://dfm.io/python-fsps/current/>`_ stellar population synthesis model
*  `emcee <https://emcee.readthedocs.io/en/stable/>`_ package for the Affine Invariant Markov Chain Monte Carlo (MCMC) Ensemble sampler
*  `Astropy <https://www.astropy.org/>`_
*  `Photutils <https://photutils.readthedocs.io/en/stable/>`_
*  `Aniano et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_ who provides convolution `kernels <https://www.astro.princeton.edu/~ganiano/Kernels.html>`_ for the PSF matching
*  `SExtractor <https://www.astromatic.net/software/sextractor>`_ (`Bertin & Arnouts 1996 <https://ui.adsabs.harvard.edu/abs/1996A%26AS..117..393B/abstract>`_)
*  Abdurro'uf & Akiyama (`2017 <https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.2806A/abstract>`_, `2018 <https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5083A/abstract>`_)
