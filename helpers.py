

import os, sys, glob, shutil

import numpy
import pyfits
from scipy.ndimage.filters import median_filter
import bottleneck
import scipy.interpolate
numpy.seterr(divide='ignore', invalid='ignore')

# Disable nasty and useless RankWarning when spline fitting
import warnings
warnings.simplefilter('ignore', numpy.RankWarning)


import pyfits
import pysalt.mp_logging
import logging
import numpy






#################################################################################
#################################################################################
#################################################################################
def get_integrated_spectrum(hdu_rect, filename):
    """

    This function integrates the spectrum along the spectral axis, returning a 
    1-D array of integrated intensities. 

    """
    
    _,fb = os.path.split(filename)
    logger = logging.getLogger("GetIntegratedSpec(%s)" % (fb))

    integrated_intensity = bottleneck.nansum(hdu_rect['SCI'].data.astype(numpy.float32), axis=1)
    logger.info("Integrated intensity: covers %d pixels along slit" % (integrated_intensity.shape[0]))
    # pyfits.PrimaryHDU(data=integrated_intensity).writeto()
    numpy.savetxt("1d_%s.cat" % (fb[:-5]), integrated_intensity)

    return integrated_intensity



#################################################################################
#################################################################################
#################################################################################
def find_slit_profile(hdulist, filename, source_region=[1400,2600]):
    """

    Starting with the intensity profile along the slit, reject all likely 
    sources (bright things), small-scale fluctuations (more sources, cosmics, 
    etc), and finally produce a spline-smoothed profile of intensity along the
    slit. This can then be used to correct the image data with the goal to 
    improve sky-line subtraction.

    """

    try:
        _,fb = os.path.split(filename)
    except:
        fb = "xxx.fits"
    logger = logging.getLogger("FindSlitProf")

    #
    # Get binning information
    #
    binx, biny = pysalt.get_binning(hdulist)
    if (binx == None):
        logger.error("Can't find binning information. Does header CCDSUM exist?")
        return None

    #
    # Get integrated slit profile
    #
    integrated_intensity = bottleneck.nansum(hdulist['SCI'].data.astype(numpy.float32), axis=1)
    logger.info("Integrated intensity: covers %d pixels along slit" % (integrated_intensity.shape[0]))
    # pyfits.PrimaryHDU(data=integrated_intensity).writeto()
    numpy.savetxt("1d_%s.cat" % (fb[:-5]), integrated_intensity)

    #
    # First of all, reject all pixels with zero fluxes
    #
    bad_pixels = (integrated_intensity <= 0) | (numpy.isnan(integrated_intensity))

    # Next, find average level across the profile.
    # That way, we hopefully can reject all bright targets
    bright_lim, faint_lim = 1e9, 0
    # background = (integrated_intensity <= bright_lim) | \
    #              (integrated_intensity >= faint_lim)
    background = ~bad_pixels
    likely_background_profile = numpy.array(integrated_intensity)
    likely_background_profile[~background] = numpy.NaN
    numpy.savetxt("1d_bg_%s.cat.start" % (fb[:-5]), likely_background_profile)

    for i in range(5):
        logger.info("Iteration %d: %d valid pixels considered BG" % (i+1, numpy.sum(background)))

        # compute median of all pixels considered background
        med = bottleneck.nanmedian(integrated_intensity[background])
        std = bottleneck.nanstd(integrated_intensity[background])
        logger.info("Med/Std: %f   %f" % (med, std))
        # Now set new bright and faint limits
        bright_lim, faint_lim = med+3*std, med-3*std

        # Apply new mask to intensities
        background = background & \
                     (integrated_intensity <= bright_lim) & \
                     (integrated_intensity >= faint_lim)

        likely_background_profile = numpy.array(integrated_intensity)
        likely_background_profile[~background] = numpy.NaN
        numpy.savetxt("1d_bg_%s.cat.%d" % (fb[:-5], i+1), likely_background_profile)

    skymask = ~bad_pixels & background

    #
    # Reject small outliers by median-filtering across a number of pixels
    #
    half_window_size = 25
    
    filtered_bg = numpy.array([
        bottleneck.nanmedian(
            likely_background_profile[i-half_window_size:i+half_window_size]) 
        for i in range(likely_background_profile.shape[0])])
    numpy.savetxt("1d_bg_%s.cat.filtered" % (fb[:-5]), filtered_bg)

    #
    # To smooth things out even better, fit a very low order spline, 
    # avoiding the central area where the source likely is located
    #
    # Use 50 basepoints for the fitting polynomial
    x = numpy.arange(filtered_bg.shape[0])
    print 400/biny, 3600/biny, 100/biny

    if (not source_region == None):
        print "X:", filtered_bg.shape
        do_not_fit = ((x<1400/biny) | (x>2600/biny)) & skymask

        t = numpy.linspace(numpy.min(x[do_not_fit])+1, numpy.max(x[do_not_fit])-1, 50) #100/biny)
        avoidance = (t>source_region[0]/biny) & (t<source_region[1]/biny)
        t = t[~avoidance]
        print t
    else:
        do_not_fit = skymask
        t = numpy.linspace(numpy.min(x)+1, numpy.max(x)-1, 50)
        
    print "Using spline fitting base points\n",t

    w = numpy.ones(filtered_bg.shape[0])
    w[~skymask] = 0
    numpy.savetxt("slitprofile.weights", w)
    numpy.savetxt("slitprofile.do_not_fit", do_not_fit)
    
    print "xrange:",numpy.min(x[do_not_fit]), numpy.max(x[do_not_fit])
    numpy.savetxt("1d_bg_%s.cat.basepoints" % (fb[:-5]), t, "%.2f")
    numpy.savetxt("1d_bg_%s.cat.wt" % (fb[:-5]), w)
    # lsq_spline = scipy.interpolate.LSQUnivariateSpline(
    #     x=x[do_not_fit], y=filtered_bg[do_not_fit], t=t, 
    #     w=None, bbox=[None, None], k=2)
    lsq_spline = scipy.interpolate.LSQUnivariateSpline(
        x=x, y=filtered_bg, t=t, 
        w=w, bbox=[None, None], k=2)
    numpy.savetxt("1d_bg_%s.cat.fit" % (fb[:-5]), lsq_spline(x))

    #
    # Also try fitting a polynomial to the function
    #
    #numpy.polyfit(

    #
    # Now normalize this line profile so we can use it to flatten out the slit image
    #
    avg_flux = bottleneck.nanmean(filtered_bg)
    print "average slit across slit profile:", avg_flux

    #slit_flattening = filtered_bg / avg_flux
    slit_flattening = lsq_spline(x) / avg_flux
    # Fill in gaps with ones
    slit_flattening[numpy.isnan(slit_flattening)] = 1.0

    return slit_flattening.reshape((-1,1)), skymask, filtered_bg.reshape((-1,1))/avg_flux


