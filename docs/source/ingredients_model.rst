Ingredients in SED modeling
===========================
In **piXedfit**, the task of generating model SEDs is done by :mod:`piXedfit_model` module. The SED modeling uses the Flexible Stellar Population Synthesis (`FSPS <https://github.com/cconroy20/fsps>`_) package through the `Python-FSPS <http://dfm.io/python-fsps/current/>`_ as the interface to the Python environment. The FSPS package provides a self-consistent modeling of galaxy's SED through a careful modeling of the physical components that make up the total luminosity output of a galaxy, which consist of stellar emission, nebular emission, dust emission, and emission from the dusty torus heated by the AGN. Since :mod:`piXedfit_model` module uses the FSPS model, every parameter (i.e., ingredient) available in the FSPS is also available in the :mod:`piXedfit_model`.

SSP model
---------
For modeling a SSP, FSPS provides several choices for the Initial Mass Function (IMF), isochrones calculation, and the stellar spectral libraries. The Chabrier et al. (2003) IMF, Padova isochrones (Girardi et al. 2000; Marigo et al. 2007; Marigo et al. 2008), and MILES stellar spectral library (Sanchez-Blazquez et al. 2006; Falcon et al. 2011} are used as the default set in the :mod:`piXedfit_model`, but in principle, all the choices available in the FSPS (python-FSPS) are also available in the :mod:`piXedfit_model`. In practice, SED fitting procedure demands model SEDs with a random set of :math:`Z` rather than in a discrete set, as given by the isochrones. In this case, we choose an option in FSPS that allows interpolation of SSP spectra between :math:`Z` grids. Users of :mod:`piXedfit_model` can choose from the 5 available choices of IMF that FSPS provides: Salpeter et al. (1955), Chabrier et al. (2003), Kroupa et al. (2001), van Dokkum et al. (2008), and Dave et al. (2008).

FSPS uses the _` CLOUDY <https://nublado.org/>`_ code (Ferland et al. 1998, Ferland et al. 2013) for the nebular emission modeling. The implementation of CLOUDY within FSPS is described in Byler et al. (2017). In short, the modeling has three parameters: SSP age, gas-phase metallicity, and the ionization parameter, :math:`U`, which represents the ratio of the ionizing photons to the total hydrogen density. By default, the gas-phase metallicity is set to be equal to the model stellar metallicity, and :math:`U` is fixed to 0.01. The user can also set them as free parameters in the fitting, preferentially if a constraining data is available (e.g., deep optical spectra). The modeling has incorporated the dust attenuation to the emission lines. 

Choices for the SFH
-------------------


Dust emission and AGN components
--------------------------------


IGM absoption, redshifting, and convolving with filters
-------------------------------------------------------
