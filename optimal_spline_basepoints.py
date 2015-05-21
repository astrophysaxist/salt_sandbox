#!/usr/bin/env python

import os, sys, pyfits, numpy
import scipy, scipy.interpolate, scipy.spatial


import pysalt.mp_logging
import logging
import bottleneck


def satisfy_schoenberg_whitney(data, basepoints, k=3):

    delete = numpy.isnan(basepoints)
    for idx in range(basepoints.shape[0]-1):
        # count how many data points are between this and the next basepoint
        in_range = (data > basepoints[idx]) & (data < basepoints[idx+1])
        count = numpy.sum(in_range)
        if (count < k):
            # delete this basepoint
            delete[idx] = True

    return basepoints[~delete]

    

def optimal_sky_subtraction(obj_hdulist, 
                            sky_regions=None,
                            slitprofile=None,
                            N_points = 6000):

    logger = logging.getLogger("OptSplineKs")

    logger.info("Loading all data from FITS")
    obj_data = obj_hdulist['SCI.RAW'].data
    obj_wl   = obj_hdulist['WAVELENGTH'].data
    obj_rms  = obj_hdulist['VAR'].data
    obj_bpm  = obj_hdulist['BPM'].data.flatten()

    try:
        obj_spatial = obj_hdulist['SPATIAL'].data
    except:
        logger.warning("Could not find spatial map, using plain x/y coordinates instead")
        obj_spatial, _ = numpy.indices(obj_data.shape)

    # now merge all data frames into a single 3-d numpy array
    obj_cube = numpy.empty((obj_data.shape[0], obj_data.shape[1], 4))
    obj_cube[:,:,0] = obj_wl[:,:]
    obj_cube[:,:,1] = obj_data[:,:]
    obj_cube[:,:,2] = obj_rms[:,:]
    obj_cube[:,:,3] = obj_spatial[:,:]
    

    obj_cube = obj_cube.reshape((-1, obj_cube.shape[2]))

    # Now exclude all pixels marked as bad
    obj_cube = obj_cube[obj_bpm == 0]

    logger.info("%7d pixels left after eliminating bad pixels!" % (obj_cube.shape[0]))

    #
    # Now also exclude all points that are marked as non-sky regions 
    # (e.g. including source regions)
    #
    if (not type(sky_regions) == type(None) and
        type(sky_regions) == numpy.ndarray):
        
        logger.info("Selecting sky-pixels from user-defined regions")
        is_sky = numpy.ones((obj_cube.shape[0]), dtype=numpy.bool)
        for idx, sky_region in enumerate(sky_regions):
            in_region = (obj_cube[:,3] > sky_region[0]) & (obj_cube[:,3] < sky_region[1])
            is_sky[in_region] = True

        obj_cube = obj_cube[is_sky]

        
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
    wl_range = wl_max - wl_min
    k_orig_ = numpy.linspace(wl_min, wl_max, N_points+2)[1:-1]
    k_orig = satisfy_schoenberg_whitney(allskies[:,0], k_orig_, k=3)
    spline_orig = scipy.interpolate.LSQUnivariateSpline(
        x=allskies[:,0], 
        y=allskies[:,1], 
        t=k_orig,
        w=None, # no weights (for now)
        #bbox=None, #[wl_min, wl_max], 
        k=3, # use a cubic spline fit
        )
    numpy.savetxt("spline_orig", numpy.append(k_orig.reshape((-1,1)),
                                              spline_orig(k_orig).reshape((-1,1)),
                                              axis=1)
                  )

    logger.info("Computing spline using optimized sampling")
    k_opt_good = satisfy_schoenberg_whitney(allskies[:,0], k_wl, k=3)
    spline_opt = scipy.interpolate.LSQUnivariateSpline(
        x=allskies[:,0], 
        y=allskies[:,1], 
        t=k_opt_good, #k_wl,
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

    good_data = allskies[good_point]

    spline_iter = None
    for iteration in range(3):

        # compute spline
        k_iter_good = satisfy_schoenberg_whitney(good_data[:,0], k_wl, k=3)

        spline_iter = scipy.interpolate.LSQUnivariateSpline(
            x=good_data[:,0], #allskies[:,0],#[good_point], 
            y=good_data[:,1], #allskies[:,1],#[good_point], 
            t=k_iter_good, #k_wl,
            w=None, # no weights (for now)
            bbox=[wl_min, wl_max], 
            k=3, # use a cubic spline fit
            )
        numpy.savetxt("spline_opt.iter%d" % (iteration+1), 
                      numpy.append(k_wl.reshape((-1,1)),
                                   spline_iter(k_wl).reshape((-1,1)),
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
            fill_value=1e3,
            bounds_error=False,
            #assume_sorted=True
            )
        var_at_pixel = std_interpol(good_data[:,0])

        numpy.savetxt("pixelvar.%d" % (iteration+1), 
                      numpy.append(good_data[:,0].reshape((-1,1)),
                                   var_at_pixel.reshape((-1,1)), axis=1))

        # Now mark all pixels exceeding the noise threshold as outliers
        not_outlier = numpy.fabs(dflux) < var_at_pixel

        good_data = good_data[not_outlier]
        numpy.savetxt("good_after.%d" % (iteration+1),
                      numpy.append(good_data[:,0].reshape((-1,1)),
                                   dflux[not_outlier].reshape((-1,1)), axis=1))

        logger.info("Done with iteration %d (%d pixels left)" % (iteration+1, good_data.shape[0]))


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


    #
    # Now in a final step, compute the 2-D sky spectrum, subtract, and save results
    #
    if (not spline_iter == None):
        sky2d = spline_iter(obj_wl.ravel()).reshape(obj_wl.shape)
        skysub = obj_data - sky2d
        ss_hdu = pyfits.ImageHDU(header=obj_hdulist['SCI.RAW'].header,
                                 data=skysub)
        ss_hdu.name = "SKYSUB.OPT"
        obj_hdulist.append(ss_hdu)


    return




if __name__ == "__main__":


    logger_setup = pysalt.mp_logging.setup_logging()


    obj_fitsfile = sys.argv[1]
    obj_hdulist = pyfits.open(obj_fitsfile)

    sky_regions = None
    if (len(sys.argv) > 2):
        user_sky = sys.argv[2]
        sky_regions = numpy.array([x.split(":") for x in user_sky.split(",")]).astype(numpy.int)


    optimal_sky_subtraction(obj_hdulist, 
                            sky_regions=sky_regions,
                            N_points=6000)

    obj_hdulist.writeto(obj_fitsfile[:-5]+".optimized.fits", clobber=True)

    pysalt.mp_logging.shutdown_logging(logger_setup)
