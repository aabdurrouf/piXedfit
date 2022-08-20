from astropy.coordinates import SkyCoord
import glob
import os, sys, shutil
import pytest

original_screenshot = glob.glob("*")
global PIXEDFIT_HOME
PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
sys.path.insert(0, PIXEDFIT_HOME)

Wasp = SkyCoord.from_name("WASP-37")
NGC = SkyCoord.from_name("NGC 309")
SN = SkyCoord.from_name("SN 2007A")

from piXedfit.piXedfit_images import images_utils as iu


@pytest.fixture
def clean_up():
    for file in glob.glob("*"):
        if file in original_screenshot:
            continue
        else:
            try:
                os.remove(file)
            except PermissionError:
                shutil.rmtree(file)
    yield


@pytest.mark.usefixtures("clean_up")
class Test_dwd():

    def test_SDSS(self):
        assert iu.dwd_sdss(Wasp, bands = ['u','z']) == 1
        assert iu.dwd_sdss(NGC, bands = ['u','z']) == 1
        assert iu.dwd_sdss(SN, bands = ['u','z']) == 1
    
    def test_2mass(self):
        assert iu.dwd_2mass(Wasp) == 1
        assert iu.dwd_2mass(NGC) == 1
        assert iu.dwd_2mass(SN) == 1
    
    def test_wise(self):
        assert iu.dwd_wise(Wasp, pix = 300) == 1
        assert iu.dwd_wise(NGC, pix = 300) == 1
        assert iu.dwd_wise(SN, pix = 300) == 1
    
    def test_galex(self):
        assert iu.dwd_galex(Wasp, maximum = 3) == 1
        assert iu.dwd_galex(NGC, maximum = 3) == 1
        assert iu.dwd_galex(SN, maximum = 3) == 1

    def test_hst(self):
        assert iu.dwd_hst(Wasp, maximum = 1) == 1
        assert iu.dwd_hst(NGC, maximum = 1) == 1
        assert iu.dwd_hst(SN, maximum = 1) == 1 
    
    def test_spitzer(self):
        assert iu.dwd_spitzer(Wasp, maximum = 1) == 1
        assert iu.dwd_spitzer(NGC, maximum = 1) == 1
        assert iu.dwd_spitzer(SN, maximum = 1) == 1

from piXedfit.piXedfit_images import var_img_sdss
from piXedfit.piXedfit_images import skybg_sdss
from astropy.io import fits
def test_sdss():
    os.chdir("tests/data/SN 2010ex")
    for i in ['u','z']:
        skybg_sdss(f"SDSS_{i}_0.fits")
        var_img_sdss(f"SDSS_{i}_0.fits",f'sdss_{i}')
    os.chdir("../")

    hdul1, hdul2 = fits.open("SN 2010ex/skybg_SDSS_u_0.fits"), fits.open("Sample/skybg_SDSS_u_0.fits")
    
    assert fits.FITSDiff(hdul1, hdul2).identical == 1
    assert fits.HDUDiff(hdul1[0], hdul2[0]).identical == 1
    assert fits.RawDataDiff(hdul1[0].data, hdul2[0].data).identical == 1
    assert fits.ImageDataDiff(hdul1[0].data, hdul2[0].data).identical == 1
    

from piXedfit.piXedfit_images import subtract_background
from piXedfit.piXedfit_images import var_img_2MASS
def test_twomass():
    os.chdir("SN 2010ex")

    file = "J_0.fits"
    fits_image = file
    # backhround subtraction
    subtract_background(fits_image, sigma=5.0, box_size=[100,100], mask_sources=True)
    
    # Deriving variance images
    sci_img = f"skybgsub_{file}"
    skyrms_img = f"skybgrms_{file}"
    name_out_fits = f"var_skybgsub_{file}"
    var_img_2MASS(sci_img=sci_img, skyrms_img=skyrms_img, name_out_fits=name_out_fits)

    os.chdir("../")

    hdul1, hdul2 = fits.open("SN 2010ex/var_skybgsub_J_0.fits"), fits.open("Sample/var_skybgsub_J_0.fits")
    assert fits.FITSDiff(hdul1, hdul2).identical == 1
    assert fits.HDUDiff(hdul1[0], hdul2[0]).identical == 1
    assert fits.RawDataDiff(hdul1[0].data, hdul2[0].data).identical == 1
    assert fits.ImageDataDiff(hdul1[0].data, hdul2[0].data).identical == 1

    hdul1, hdul2 = fits.open("SN 2010ex/skybg_J_0.fits"), fits.open("Sample/skybg_J_0.fits")
    assert fits.FITSDiff(hdul1, hdul2).identical == 1
    assert fits.HDUDiff(hdul1[0], hdul2[0]).identical == 1
    assert fits.RawDataDiff(hdul1[0].data, hdul2[0].data).identical == 1
    assert fits.ImageDataDiff(hdul1[0].data, hdul2[0].data).identical == 1


from piXedfit.piXedfit_images import var_img_WISE
def test_wise():
    os.chdir("SN 2010ex")
    for suffix in ['W1','W4']:
        file = f"{suffix}_0.fits"
        fits_image = file
        
        # backhround subtraction
        subtract_background(fits_image, sigma=5.0, box_size=[100,100], mask_sources=True)
        
        # Deriving variance images
        sci_img = f"skybgsub_{file}"
        skyrms_img = f"skybgrms_{file}"
        unc_img = f"{suffix}_unc_0.fits"
        name_out_fits = f"var_skybgsub_{file}"
        print(name_out_fits)
        var_img_WISE(sci_img=sci_img, unc_img = unc_img, filter_name = f"wise_{suffix.lower()}" ,skyrms_img=skyrms_img, name_out_fits=name_out_fits)
    
    os.chdir("../")

    hdul1, hdul2 = fits.open("SN 2010ex/skybgsub_W1_0.fits"), fits.open("Sample/skybgsub_W1_0.fits")
    assert fits.FITSDiff(hdul1, hdul2).identical == 1
    assert fits.HDUDiff(hdul1[0], hdul2[0]).identical == 1
    assert fits.RawDataDiff(hdul1[0].data, hdul2[0].data).identical == 1
    assert fits.ImageDataDiff(hdul1[0].data, hdul2[0].data).identical == 1 

    hdul1, hdul2 = fits.open("SN 2010ex/skybgsub_W4_0.fits"), fits.open("Sample/skybgsub_W4_0.fits")
    assert fits.FITSDiff(hdul1, hdul2).identical == 1
    assert fits.HDUDiff(hdul1[0], hdul2[0]).identical == 1
    assert fits.RawDataDiff(hdul1[0].data, hdul2[0].data).identical == 1
    assert fits.ImageDataDiff(hdul1[0].data, hdul2[0].data).identical == 1

    os.chdir("../")
    os.chdir("../")