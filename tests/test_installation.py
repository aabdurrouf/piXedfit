import importlib.util as ut
import os

def test_installlation():
    assert ut.find_spec('piXedfit') != None , "piXedfit is not installed as a package."
    
def test_env_path():
    assert os.environ['PIXEDFIT_HOME'] != KeyError, "Please make sure piXedfit is in your system env."
    
def test_FSPS():
    assert os.environ['SPS_HOME'] != KeyError, "Please make sure FSPS is in your system env."
    
def test_fsps():
    assert ut.find_spec('fsps') != None , "fsps is not installed."
    
def test_mpi4py():
    assert ut.find_spec('mpi4py') != None , "mpi4py is not installed."
    
def test_emcee():
    assert ut.find_spec('emcee') != None , "emcee is not installed. You may be unable to run MCMC."
    
def test_h5py():
    assert ut.find_spec('h5py') != None , "h5py is not installed."
    

