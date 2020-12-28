Ingredients in SED modeling
===========================
In **piXedfit**, the task of generating model SEDs is done by :mod:`piXedfit_model` module. The SED modeling uses the Flexible Stellar Population Synthesis (`FSPS <https://github.com/cconroy20/fsps>`_) package through the `Python-FSPS <http://dfm.io/python-fsps/current/>`_ as the interface to the Python environment. The FSPS package provides a self-consistent modeling of galaxy's SED through a careful modeling of the physical components that make up the total luminosity output of a galaxy, which consist of stellar emission, nebular emission, dust emission, and emission from the dusty torus heated by the AGN. Since :mod:`piXedfit_model` module uses the FSPS model, every parameter (i.e., ingredient) available in the FSPS is also available in the :mod:`piXedfit_model`.

SSP model
---------
For modeling a Simple Stellar Population (SSP), the FSPS provides several choices for the Initial Mass Function (IMF), isochrones calculation, and the stellar spectral libraries. The Chabrier et al. (2003) IMF, Padova isochrones (Girardi et al. 2000; Marigo et al. 2007; Marigo et al. 2008), and MILES stellar spectral library (Sanchez-Blazquez et al. 2006; Falcon et al. 2011} are used as the default set in the :mod:`piXedfit_model`, but in principle, all the choices available in the FSPS (python-FSPS) are also available in the :mod:`piXedfit_model`. In practice, SED fitting procedure demands model SEDs with a random set of :math:`Z` rather than in a discrete set, as given by the isochrones. In this case, we choose an option in FSPS that allows interpolation of SSP spectra between :math:`Z` grids. Users of :mod:`piXedfit_model` can choose from the 5 available choices of IMF that FSPS provides: Salpeter et al. (1955), Chabrier et al. (2003), Kroupa et al. (2001), van Dokkum et al. (2008), and Dave et al. (2008).

FSPS uses the `CLOUDY <https://nublado.org/>`_ code (Ferland et al. 1998, Ferland et al. 2013) for the nebular emission modeling. The implementation of CLOUDY within FSPS is described in Byler et al. (2017). In short, the modeling has three parameters: SSP age, gas-phase metallicity, and the ionization parameter, :math:`U`, which represents the ratio of the ionizing photons to the total hydrogen density. By default, the gas-phase metallicity is set to be equal to the model stellar metallicity, and :math:`U` is fixed to 0.01. The user can also set them as free parameters in the fitting, preferentially if a constraining data is available (e.g., deep optical spectra). The modeling has incorporated the dust attenuation to the emission lines.

There are five options for the dust attenuation modeling in FSPS. We only accommodate two of them in :mod:`piXedfit_model`: Calzetti et al. (2000) and the two-component Charlot & Fall (2000) dust attenuation model. In brief, the Calzetti et al. (2000) assumes equal dust attenuation over all starlight regardless of the stellar ages, while Charlot & Fall (2000) assumes an extra attenuation for the light coming from young stars (typically younger than $10$ Myr) which still reside in the birth-cloud. For the Calzetti et al. (2000) dust attenuation model, only one parameter is involved, :math:`\hat{\tau}_{2}` which represents the dust optical depth. For the two-component Charlot & Fall (2000) model, there are three parameters involved: (1) :math:`\hat{\tau}_{1}` controls normalization of the attenuation curve for the birth-cloud component, (2) :math:`\hat{\tau}_{2}` controls the normalization of the attenuation curve for the diffuse interstellar medium (ISM) component, and (3) the power-law index :math:`n` in the dust attenuation curve for the diffuse component (see Eq. 7 and 8 in Leja et al. 2017). 


Choices for the SFH
-------------------
**piXedfit** adopts the parametric star formation history (SFH) approach, which assumes a functional form for the SFH when generating the model SED of a Composite Stellar Population (CSP). In :mod:`piXedfit_model`, there are 5 choices of SFH available:     

* **Tau model**
  :math:`SFR(t) \propto e^{-t/\tau}`
  
  The :math:`\tau` represents the timescale for the declining of the star formation. 

* **Delayed tau**
  :math:`SFR(t) \propto t e^{-t/\tau}`
  
  The :math:`\tau` is a parameter that controls the duration of the star formation.

* **Log-normal**
  :math:`SFR(t) \propto \frac{1}{t} \exp \left( -\frac{(\ln(t)-T_{0})^{2}}{2\tau^{2}} \right)`
  
  The free parameters :math:`T_{0}` controls the peak location, while :math:`\tau` controls the duration of the star formation.

* **Gaussian**
  :math:`SFR(t) \propto \exp \left( -\frac{(t-T_{0})^{2}}{2\tau^{2}} \right)`
  
  The :math:`T_{0}` represents the time when star formation reaches the peak, while the :math:`\tau` controls the duration of the star formation.  

* **Double power law**
  :math:`SFR(t) \propto \left[ \left(\frac{t}{\tau} \right)^{\alpha} + \left(\frac{t}{\tau} \right)^{-\beta} \right]^{-1}`    
  
  The :math:`\alpha` and :math:`\beta` are the falling slope, and the rising slope, respectively. The :math:`\tau` parameter controls the peak time. 

All the :math:`t` in the above equations represent the time since the start of star formation (i.e., age of the system, :math:`\text{age}_{\text{sys}}`).
The following figure shows examples of SFHs formed with the 5 SFH choices. All the model SFHs have the same age :math:`t` of 12.5 Gyr and :math:`M_{*}=5.0\times 10^{10}M_{\odot}`. The other SFH parameters are: tau model [:math:`\tau=4.0` Gyr], delayed tau [:math:`\tau=2.5` Gyr], log-normal [:math:`\tau=1.0` Gyr, :math:`T_{0}=1.3` Gyr], Gaussian [:math:`\tau=2.5` Gyr, :math:`T_{0}=7.0` Gyr], and double power law [:math:`\tau=2.5` Gyr, :math:`\alpha=2.0` Gyr, :math:`\beta=2.0` Gyr].

.. image:: paramSFH.png


Dust emission and AGN components
--------------------------------


IGM absoption, redshifting, and convolving with filters
-------------------------------------------------------
