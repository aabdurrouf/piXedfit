piXedfit
========
**piXefit** provides a compehensive set of tools for analyzing spatially resolved spectral energy distributions (SEDs) of galaxies and dissecting the spatially resolved properties of the stellar populations and dust in the galaxies. First, it can produce a pixel-matched 3D data cube from an input of a set of mutliband imaging data alone or in combination with an integral field spectroscopy (IFS) data. When IFS data is provided, it can produce a 3D spectrophotometric data cube in which spectra and photometric SEDs are combined on pixel level. Second, it has a unique pixel binning feature that can optimize the S/N ratio of SEDs on spatially resolved scales while retaining the spatial and spectral variations of the SEDs by accounting the similarity of SED shape of pixels in the binning process. This can be expected to reduce biases introduced by the binning process that combines pixels regardless of the variations in their SED shapes. Finally, piXedfit also provides a stand-alone SED fitting capability. It has two options of fitting methods: MCMC and random dense sampling of parameter space (RDSPS). Most of the modules in piXedfit have implemented MPI for parallel computation. A detailed description of piXedfit is presented in `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_. Some examples of practical usages and tutorials can be found at `examples <https://github.com/aabdurrouf/piXedfit/tree/main/examples>`_. 

.. image:: 3Dcube_specphoto.png
.. image:: demo_pixedfit_ngc309_new_edit.svg
.. image:: plot_maps_props_new.svg
   :width: 600

Features
--------
**piXedfit** has six modules that work independently with each other. For instance, it is possible to use the SED fitting module for fitting either global (integrated) or spatially resolved SEDs of galaxies. Those modules include:

*  :mod:`piXedfit_images`: image processing.
*  :mod:`piXedfit_spectrophotometric`: spatial and spectral matching between multiband imaging data and IFS data.   
*  :mod:`piXedfit_bin`: pixel binning to optimize S/N of SEDs on spatially resolved scales.  
*  :mod:`piXedfit_model`: generating model SEDs.     
*  :mod:`piXedfit_fitting`: SED fitting on spatially resolved scales or global (integrated) scales.
*  :mod:`piXedfit_analysis`: Analysis of SED fitting result, including visualization plots and retrieving best-fitting parameters.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   install
   manage_filters
   list_imaging_data
   list_kernels_psf
   ingredients_model
   gen_model_SEDs
   image_pros
   image_ifs_match
   pixel_binning
   fit_ing_SEDs
   fit_res_SEDs
   plot_fitres
   get_maps_prop


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials


.. toctree::
   :maxdepth: 2
   :caption: Demonstrations
   
   demos_pixel_binning
   demos_sed_fitting


.. toctree::
   :maxdepth: 2
   :caption: API reference 

   piXedfit_images
   piXedfit_spectrophotometric
   piXedfit_bin
   piXedfit_model
   piXedfit_fitting
   piXedfit_analysis
   utils

Citation
--------
If you use this code for your research, please reference `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...15A/abstract>`_. If you use the pixel binning module (:mod:`piXedfit_bin`), please also reference `Abdurro'uf & Akiyama (2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.2806A/abstract>`_.


Reference
---------

A list of a few projects **piXedfit** is benefited from:

*  `Astropy <https://www.astropy.org/>`_
*  `Photutils <https://photutils.readthedocs.io/en/stable/>`_
*  `Aniano et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011PASP..123.1218A/abstract>`_ who provides convolution `kernels <https://www.astro.princeton.edu/~ganiano/Kernels.html>`_ for the PSF matching
*  `FSPS <https://github.com/cconroy20/fsps>`_ and `Python-FSPS <http://dfm.io/python-fsps/current/>`_ stellar population synthesis model
*  `emcee <https://emcee.readthedocs.io/en/stable/>`_ package for the Markov Chain Monte Carlo (MCMC) Ensemble sampler
