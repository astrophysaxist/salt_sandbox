#!/usr/bin/env python

import os, sys, pyfits, numpy
import scipy, scipy.interpolate, scipy.spatial


import pysalt.mp_logging
import logging
import bottleneck


    #     #
    #     # For debugging, compute the spline fit at all basepoints and dump to txt file
    #     #
    #     ss = numpy.append(basepoints.reshape((-1,1)),
    #                       sky_spectrum_spline(basepoints).reshape((-1,1)),
    #                       axis=1)
    #     numpy.savetxt("skyspectrum.knots", sky_spectrum_spline.get_knots())
    #     numpy.savetxt("skyspectrum.coeffs", sky_spectrum_spline.get_coeffs())
    #     numpy.savetxt("skyspectrum.txt", ss)
    # except:
    #     logger.critical("Error with spline-fitting the sky-spectrum")
    #     pysalt.mp_logging.log_exception()


if __name__ == "__main__":


    logger_setup = pysalt.mp_logging.setup_logging()
    logger = logging.getLogger("OptSplineKs")


    obj_fitsfile = sys.argv[1]
    obj_hdulist = pyfits.open(obj_fitsfile)

    logger.info("Loading all data from FITS")
    obj_data = obj_hdulist['SCIRAW'].data
    obj_wl   = obj_hdulist['WAVELENGTH'].data
    obj_rms  = obj_hdulist['VAR'].data
    obj_bpm  = obj_hdulist['BPM'].data.flatten()

    # now merge all data frames into a single 3-d numpy array
    obj_cube = numpy.empty((obj_data.shape[0], obj_data.shape[1], 3))
    obj_cube[:,:,0] = obj_wl[:,:]
    obj_cube[:,:,1] = obj_data[:,:]
    obj_cube[:,:,2] = obj_rms[:,:]

    obj_cube = obj_cube.reshape((-1, 3))

    # Now exclude all pixels marked as bad
    obj_cube = obj_cube[obj_bpm == 0]

    logger.info("%7d pixels left after eliminating bad pixels!" % (obj_cube.shape[0]))

    allskies = obj_cube[::5]

    
    

    # #
    # # Load and prepare data
    # #
    # allskies = numpy.loadtxt(allskies_filename)

    # just to be on the safe side, sort allskies by wavelength
    sky_sort_wl = numpy.argsort(allskies[:,0])
    allskies = allskies[sky_sort_wl]

    logger.debug("Working on %7d data points" % (allskies.shape[0]))


    #
    # Compute cumulative distribution
    #
    logger.info("Computing cumulative distribution")
    allskies_cumulative = numpy.cumsum(allskies[:,1], axis=0)

    # print allskies.shape, allskies_cumulative.shape, wl_sorted.shape

    numpy.savetxt("cumulative.asc", 
                  numpy.append(allskies[:,0].reshape((-1,1)),
                               allskies_cumulative.reshape((-1,1)),
                               axis=1)
                  )

    #############################################################################
    #
    # Now create the basepoints by equally distributing them across the 
    # cumulative distribution. This  naturally puts more basepoints into regions
    # with more signal where more precision is needed
    #
    #############################################################################

    # Create a simple interpolator to make life a bit easier
    interp = scipy.interpolate.interp1d(
        x=allskies_cumulative,
        y=allskies[:,0],
        kind='nearest',
        #assume_sorted=True,
        )
    
    # now create the raw basepoints in cumulative flux space
    N_points = 1000
    k_cumflux = numpy.linspace(allskies_cumulative[0],
                               allskies_cumulative[-1],
                               N_points+2)[1:-1]
    
    # and using the interpolator, convert flux space into wavelength
    k_wl = interp(k_cumflux)

    numpy.savetxt("opt_basepoints", 
                  numpy.append(k_wl.reshape((-1,1)),
                               k_cumflux.reshape((-1,1)),
                               axis=1)
                  )

    #############################################################################
    #
    # Now we have the new optimal set of base points, let's compare it to the 
    # original with the same number of basepoints, sampling the available data
    # with points equi-distant in wavelength space.
    #
    #############################################################################

    wl_min, wl_max = numpy.min(allskies[:,0]), numpy.max(allskies[:,0])

    logger.info("Computing spline using original/simple sampling")
    k_orig = numpy.linspace(wl_min, wl_max, N_points+2)[1:-1]
    spline_orig = scipy.interpolate.LSQUnivariateSpline(
        x=allskies[:,0], 
        y=allskies[:,1], 
        t=k_orig,
        w=None, # no weights (for now)
        bbox=[wl_min, wl_max], 
        k=3, # use a cubic spline fit
        )
    numpy.savetxt("spline_orig", numpy.append(k_orig.reshape((-1,1)),
                                              spline_orig(k_orig).reshape((-1,1)),
                                              axis=1)
                  )

    logger.info("Computing spline using optimized sampling")
    spline_opt = scipy.interpolate.LSQUnivariateSpline(
        x=allskies[:,0], 
        y=allskies[:,1], 
        t=k_wl,
        w=None, # no weights (for now)
        bbox=[wl_min, wl_max], 
        k=3, # use a cubic spline fit
        )
    numpy.savetxt("spline_opt", numpy.append(k_wl.reshape((-1,1)),
                                             spline_opt(k_wl).reshape((-1,1)),
                                             axis=1)
                  )

    
    logger.info("Computing spline using optimized sampling and outlier rejection")
    good_point = (allskies[:,0] > 0)
    print good_point.shape
    print good_point

    avg_sample_width = (numpy.max(k_wl) - numpy.min(k_wl)) / k_wl.shape[0]

    for iteration in range(3):

        good_data = allskies[good_point]

        # compute spline
        spline_iter = scipy.interpolate.LSQUnivariateSpline(
            x=allskies[:,0],#[good_point], 
            y=allskies[:,1],#[good_point], 
            t=k_wl,
            w=None, # no weights (for now)
            bbox=[wl_min, wl_max], 
            k=3, # use a cubic spline fit
            )
        numpy.savetxt("spline_opt.iter%d" % (iteration+1), 
                      numpy.append(k_wl.reshape((-1,1)),
                                   spline_opt(k_wl).reshape((-1,1)),
                                   axis=1)
                  )
        
        
        # compute spline fit for each wavelength data point
        dflux = good_data[:,1] - spline_iter(good_data[:,0])
        print dflux

        #
        # Add here: work out the scatter of the distribution of pixels in the 
        #           vicinity of this basepoint. This is what determined outlier 
        #           or not, and NOT the uncertainty in a given pixel
        #

        # Create a KD-tree with all data points
        wl_tree = scipy.spatial.cKDTree(good_data[:,0].reshape((-1,1)))

        # Now search this tree for points near each of the spline base points
        d, i = wl_tree.query(k_wl.reshape((-1,1)),
                             k=100, # use 100 neighbors
                             distance_upper_bound=avg_sample_width)
        
        # make sure to flag outliers
        bad = (i >= dflux.shape[0])
        i[bad] = 0

        # Now we have all indices of a bunch of nearby datapoints, so we can 
        # extract how far off each of the data points is
        delta_flux_2d = dflux[i]
        delta_flux_2d[bad] = numpy.NaN
        print "dflux_2d = ", delta_flux_2d.shape

        # With this we can estimate the scatter around each spline fit basepoint
        var = bottleneck.nanstd(delta_flux_2d, axis=1)
        print "variance:", var.shape
        numpy.savetxt("fit_variance.iter_%d" % (iteration+1),
                      numpy.append(k_wl.reshape((-1,1)),
                                   var.reshape((-1,1)), axis=1))
        
        #
        # Now interpolate this scatter linearly to the position of each 
        # datapoint in the original dataset. That way we can easily decide, 
        # for each individual pixel, if that pixel is to be considered an 
        # outlier or not.
        #
        # Note: Need to consider ALL pixels here, not just the good ones 
        #       selected above
        #
        std_interpol = scipy.interpolate.interp1d(
            x = k_wl, 
            y = var,
            kind = 'linear',
            #assume_sorted=True
            )
        var_at_pixel = std_interpol(good_data[:,0])

        # Now mark all pixels exceeding the noise threshold as outliers
        not_outlier = numpy.fabs(dflux) < var_at_pixel

        good_data = good_data[not_outlier]


        # # and check how significant this deviation is
        # dsigma = dflux / allskies[:,2]
        # #print dsigma

        
        # # now exclude all data points that deviate by more than 3 sigma
        # good_point = numpy.fabs(dsigma) < 3
        # #print good_point.shape
        # #print good_point

        logger.info("Done with iteration %d (%d pixels left)" % (iteration+1, good_data.shape[0]))
#numpy.sum(good_point)))


    #
    # Compute differences between data and spline fit for both cases
    #
    logger.info("computing comparison data")
    fit_orig = spline_orig(allskies[:,0])
    fit_opt = spline_opt(allskies[:,0])
    
    comp = numpy.zeros((allskies.shape[0], allskies.shape[1]+2))
    comp[:, :allskies.shape[1]] = allskies[:,:]
    comp[:, allskies.shape[1]+0] = fit_orig[:]
    comp[:, allskies.shape[1]+1] = fit_opt[:]
    numpy.savetxt("allskies.comp", comp)

    pysalt.mp_logging.shutdown_logging(logger_setup)
