import numpy as np
import math
import sys, os
from astropy.cosmology import *
from astropy.constants import L_sun

__all__ = ["cosmo_redshifting"]

def cosmo_redshifting(wave=[], spec=[], z=0.01, cosmo='flat_LCDM', H0=70.0, Om0=0.3, DL_Gpc=0.0):
    
    if DL_Gpc > 0.0:
        D_L_cm = DL_Gpc*3.08568e+27    # in cm
    else:
        if cosmo=='flat_LCDM' or cosmo==0:
            cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
            DL = cosmo1.luminosity_distance(z)      # in unit of Mpc
        elif cosmo=='WMAP5' or cosmo==1:
            DL = WMAP5.luminosity_distance(z)
        elif cosmo=='WMAP7' or cosmo==2:
            DL = WMAP7.luminosity_distance(z)
        elif cosmo=='WMAP9' or cosmo==3:
            DL = WMAP9.luminosity_distance(z)
        elif cosmo=='Planck13' or cosmo==4:
            DL = Planck13.luminosity_distance(z)
        elif cosmo=='Planck15' or cosmo==5:
            DL = Planck15.luminosity_distance(z)
        
        D_L_cm = DL.value*3.08568e+24

    # Observed wavelength (redshifted)
    wave_obs = wave * (1 + z)
    L_lambda_erg = spec * L_sun.to('erg/s').value  # Now in erg/s/Angstrom
    F_lambda_obs = (L_lambda_erg / (4 * np.pi * D_L_cm**2)) / (1 + z)  # in erg/s/cm^2/Angstrom

    return wave_obs, F_lambda_obs


def cosmo_redshifting_old(DL_Gpc=0.0,cosmo='flat_LCDM',H0=70.0,Om0=0.3,z=0.01,wave=[],spec=[]):
    """
    :param DL_Gpc (default: 0.0):
        Luminosity distance (LD) in unit of Gpc. If this parameter is not zero, the LD
        will not be calculated using the Astropy Cosmology package. 

    :param cosmo (default: 'flat_LCDM'):
        Choices for the cosmological parameters. The choices are: ['flat_LCDM', 'WMAP5', 
        'WMAP7', 'WMAP9', 'Planck13', 'Planck15'], similar to the choices available in the 
        Astropy Cosmology package: https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies.
        If 'flat_LCDM' is chosen, the input H0 and Om0 should be provided.

    :param H0 (default: 70.0):
        Hubble constant at z=0.

    :param Om0 (default: 0.3):
        Omega matter at z=0.

    :param z (default: 0.01):
        Redshift.

    :param wave:
        Wavelength grids of the input spectrum.

    :param spec:
        Fluxes of the input spectrum.  

    :returns redsh_wave:
        Wavelength grids of the redshifted spectrum

    :returns redsh_spec:
        Fluxes of redshifted spectrum. 
    """

    if DL_Gpc > 0.0:
        DL = DL_Gpc
        DL = DL*3.08568e+27
    else:
        if cosmo=='flat_LCDM' or cosmo==0:
            cosmo1 = FlatLambdaCDM(H0=H0, Om0=Om0)
            DL = cosmo1.luminosity_distance(z)      # in unit of Mpc
        elif cosmo=='WMAP5' or cosmo==1:
            DL = WMAP5.luminosity_distance(z)
        elif cosmo=='WMAP7' or cosmo==2:
            DL = WMAP7.luminosity_distance(z)
        elif cosmo=='WMAP9' or cosmo==3:
            DL = WMAP9.luminosity_distance(z)
        elif cosmo=='Planck13' or cosmo==4:
            DL = Planck13.luminosity_distance(z)
        elif cosmo=='Planck15' or cosmo==5:
            DL = Planck15.luminosity_distance(z)
        #elif cosmo=='Planck18' or cosmo==6:
        #    DL = Planck18.luminosity_distance(z)
        
        DL = DL.value/1.0e+3
        DL = DL*3.08568e+27

    redsh_wave = (1.0+z)*np.asarray(wave)
    cor = 1.0/12.56637061/DL/DL/(1.0+z)           # flux in L_solar cm^-2 A^-1
    cor = cor*3.826e+33                           # flux in erg s^-1 cm^-2 A^-1
    redsh_spec = cor*np.asarray(spec)

    return redsh_wave, redsh_spec

