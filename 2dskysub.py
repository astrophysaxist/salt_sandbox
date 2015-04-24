#!/usr/bin/env python


import os, sys, pyfits
import wlcal
import traceline

import pysalt.mp_logging
import logging
import numpy
import scipy, scipy.interpolate

if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("MAIN")


    arcfile = sys.argv[1]
    objfile = sys.argv[2]
    logger.info("Extracting WL solution from %s, applying to %s" % (
            arcfile, objfile))

    logger.info("Computing 2-D wavelength map")
    wls_2d = traceline.compute_2d_wavelength_solution(
        arc_filename=arcfile, 
        n_lines_to_trace=10, 
        fit_order=2,
        output_wavelength_image="wl+image.fits",
        debug=False)

    #
    # Now we should have a full 2-D wavelength model for our data frame
    #
    obj_hdulist = pyfits.open(objfile)
    
    obj_out = pyfits.HDUList([
            pyfits.PrimaryHDU(),
            pyfits.ImageHDU(header=obj_hdulist['SCI'].header,
                            data=obj_hdulist['SCI'].data),
            pyfits.ImageHDU(data=wls_2d),
            ])
    obj_out.writeto(sys.argv[3], clobber=True)

    #
    # Now extract some sky-spectrum from the specified y-range 
    #

    obj_data = obj_hdulist['SCI'].data

    user_sky = sys.argv[4]
    sky_regions = numpy.array([x.split(":") for x in user_sky.split(",")]).astype(numpy.int)

    # Remember: Both FITS data __AND__ WLS_2D data are in [y,x] ordering
    all_skies = None

    obj_masked = numpy.empty(obj_data.shape)
    obj_masked[:,:] = numpy.NaN

    for idx, sky_region in enumerate(sky_regions):
        print sky_region

        print obj_data.shape,  sky_region[0], sky_region[1]
        data_region = obj_data[sky_region[0]:sky_region[1], :]
        wls_region = wls_2d[sky_region[0]:sky_region[1], :]
        
        obj_masked[sky_region[0]:sky_region[1], :] = obj_data[sky_region[0]:sky_region[1], :]

        # Now merge data and wavelengths
        # this gives us a 2-D array, shape N,2 with WL in the zero-th column, 
        # and fluxes in the first
        print data_region.shape, wls_region.shape
        data_wls = numpy.append(wls_region.reshape((-1,1)),
                                data_region.reshape((-1,1)),
                                axis=1)
        # For now dump this data to file
        if (idx == 0):
            numpy.savetxt("wl+data__%d-%d.dump" % (sky_region[0],sky_region[1]),
                          data_wls)

        all_skies = data_wls if all_skies == None else \
            numpy.append(all_skies, data_wls, axis=0)

    pyfits.HDUList([pyfits.PrimaryHDU(header=obj_hdulist['SCI'].header,
                                      data=obj_masked)]).writeto("obj_masked.fits", clobber=True)

    #
    # Exclude all points with NaNs in either wavelength or flux
    #
    good_pixel = numpy.isfinite(all_skies[:,0]) & numpy.isfinite(all_skies[:,1])
    all_skies = all_skies[good_pixel]
    
    #
    # also sort all pixels to be ascending in wavelength, otherwise the spline 
    # fitting will crap out with some "Interior knots t must satisfy "
    # Schoenberg-Whitney conditions" error message that does not seem to make 
    # any sense
    #
    wl_sort = numpy.argsort(all_skies[:,0])
    all_skies = all_skies[wl_sort]

    numpy.savetxt("allskies", all_skies[::10])

    ############################################################################
    #
    # Now we have a full list of wavelengths and presumed sky fluxes
    #
    ############################################################################

    #
    # Find minimum and maximum wavelength range so we can compute 
    # a spline fit to the sky spectrum
    #
    wl_min = numpy.min(all_skies[:,0])
    wl_max = numpy.max(all_skies[:,0])
    logger.info("Sky wavelength range: %f -- %f" % (wl_min, wl_max))

    #
    # Fit a spline to the spectrum. Use N times as many basepoints as there 
    # are pixels in spectral direction in the original FITS data
    #
    N_oversample = 2.
    N_original = obj_data.shape[1]
    logger.info("Oversampling %d input pixels by a factor of %.1f" % (
            N_original, N_oversample))
    # compute basepoints
    # skip first and last to ensure we do not exceed the input range
    basepoints = numpy.linspace(wl_min, wl_max, N_original*N_oversample+2)[1:-1]
    logger.info("Using basepoints in range %f -- %f for spline fit" % (
            basepoints[0], basepoints[-1]))

    # -- For debugging --
    numpy.save("spline_x", all_skies[:,0])
    numpy.save("spline_y", all_skies[:,1])
    numpy.savetxt("spline_t", basepoints)
    
    #
    # Now attempt the actual spline fit
    #
    try:
        sky_spectrum_spline = scipy.interpolate.LSQUnivariateSpline(
            x=all_skies[:,0], 
            y=all_skies[:,1], 
            t=basepoints, 
            w=None, # no weights (for now)
            bbox=[wl_min, wl_max], 
            k=3, # use a cubic spline fit
            )

        #
        # For debugging, compute the spline fit at all basepoints and dump to txt file
        #
        ss = numpy.append(basepoints.reshape((-1,1)),
                          sky_spectrum_spline(basepoints).reshape((-1,1)),
                          axis=1)
        numpy.savetxt("skyspectrum.knots", sky_spectrum_spline.get_knots())
        numpy.savetxt("skyspectrum.coeffs", sky_spectrum_spline.get_coeffs())
        numpy.savetxt("skyspectrum.txt", ss)
    except:
        logger.critical("Error with spline-fitting the sky-spectrum")
        pysalt.mp_logging.log_exception()

        pass

    #
    # Now with the spline fit to the sky-spectrum, we can compute the 2-D sky 
    # spectrum for the full input frame, including the curvature in the spectral 
    # dimension which we haven't compensated for yet.
    #
    logger.info("Computing full-frame, 2-D sky spectrum, incl. curvature ...")
    sky_2d = sky_spectrum_spline(wls_2d.ravel())
    sky_2d = sky_2d.reshape(wls_2d.shape)
    
    # For now, write the sky spectrum to FITS so we can have a look at it in ds9
    pyfits.HDUList([pyfits.PrimaryHDU(data=sky_2d)]).writeto("sky_2d.fits", clobber=True)


    #
    # Perform the sky-subtraction (this is now easy as pie)
    #
    skysub_data = obj_data - sky_2d
    pyfits.HDUList([pyfits.PrimaryHDU(data=skysub_data)]).writeto("skysub_2d.fits", clobber=True)

    #numpy.array(sys.argv[4].split(",")).astype(numpy.int)

    pysalt.mp_logging.shutdown_logging(logger_setup)
