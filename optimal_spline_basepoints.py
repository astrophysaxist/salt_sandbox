#!/usr/bin/env python

import os, sys, numpy, time
import scipy, scipy.interpolate, scipy.spatial, scipy.ndimage

from astropy.io import fits

import pysalt.mp_logging
import logging
import bottleneck
import wlcal
import skyline_intensity
import logging
import find_edges_of_skylines
import fastedge
import skytrace
import wlmodel


use_fast_edges = True

def satisfy_schoenberg_whitney(data, basepoints, k=3):

    logger = logging.getLogger("SchoenbergWhitney")

    logger.debug("Starting with %d basepoints" % (basepoints.shape[0]))

    delete = numpy.isnan(basepoints)
    for idx in range(basepoints.shape[0]-1):
        # count how many data points are between this and the next basepoint
        in_range = (data > basepoints[idx]) & (data < basepoints[idx+1])
        count = numpy.sum(in_range)
        if (count <= k):
            # delete this basepoint
            delete[idx] = True
            logger.debug("BP % 5d: Deleting basepoint @ %.5f (idx, %d data points, < %d)" % (
                idx, basepoints[idx], count, k))

    logger.debug("Deleting %d basepoints, left with %d" % (
        numpy.sum(delete), basepoints.shape[0]-numpy.sum(delete)))

    return basepoints[~delete]

    
def find_source_mask(img_data):

    #
    # Flatten image in wavelength direction
    #
    flat = bottleneck.nanmedian(img_data.astype(numpy.float32), axis=1)
    print img_data.shape, flat.shape

    numpy.savetxt("obj_mask.flat", flat)

    median_level = numpy.median(flat)
    print median_level


    # do running median filter
    med_filt = scipy.ndimage.filters.median_filter(flat.reshape((-1,1)), size=49, mode='mirror')[:,0]
    numpy.savetxt("obj_mask.medfilt", med_filt)

    excess = flat - med_filt
    good = numpy.isfinite(excess, dtype=numpy.bool)
    print good
    print numpy.sum(good)

    combined = numpy.append(numpy.arange(excess.shape[0]).reshape((-1,1)),
                            excess.reshape((-1,1)), axis=1)
    # compute noise

    for i in range(3):
        _med = numpy.median(excess[good])
        _std = numpy.std(excess[good])
        print _med, _std
        good = (excess > _med-3*_std) & (excess < _med+3*_std)
        print numpy.sum(good)
        numpy.savetxt("obj_mask.filter%d" % (i+1), combined[good])
        
    source = ~good
    print source

    source_mask = scipy.ndimage.filters.convolve(
        input=source, 
        weights=numpy.ones((11)), 
        output=None, 
        mode='reflect', cval=0.0)
    print source_mask
    
    numpy.savetxt("obj_mask.src", combined[source_mask])

    return source_mask

    pass



def optimal_sky_subtraction(obj_hdulist, 
                            sky_regions=None,
                            slitprofile=None,
                            N_points = 6000,
                            compare=False,
                            iterate=False,
                            return_2d = True,
                            skiplength=1,
                            mask_objects=True,
                            add_edges=True,
                            skyline_flat=None,
                            select_region=None):

    logger = logging.getLogger("OptSplineKs")
    skiplength = 1

    #
    # Prepare a new refined wavelength map by using sky-lines
    #
    (x_eff, wl_map, medians, p_scale, p_skew, fm) = skytrace.create_wlmap_from_skylines(obj_hdulist)

    wlmap_model = wlmodel.rssmodelwave(
        header=obj_hdulist[0].header, 
        img=obj_hdulist['SCI'].data,
        xbin=4, ybin=4)

    logger.info("Loading all data from FITS")
    obj_data = obj_hdulist['SCI.RAW'].data #/ fm.reshape((-1,1))
    obj_wl   = wlmap_model #wl_map #obj_hdulist['WAVELENGTH'].data
    obj_rms  = obj_hdulist['VAR'].data / fm.reshape((-1,1))

    pysalt.clobberfile("XXX.fits")
    obj_hdulist.writeto("XXX.fits", clobber=True)

    try:
        obj_spatial = obj_hdulist['SPATIAL'].data
    except:
        logger.warning("Could not find spatial map, using plain x/y coordinates instead")
        obj_spatial, _ = numpy.indices(obj_data.shape)

    # now merge all data frames into a single 3-d numpy array
    obj_cube = numpy.empty((obj_data.shape[0], obj_data.shape[1], 4))
    obj_cube[:,:,0] = obj_wl[:,:]
    obj_cube[:,:,1] = (obj_data*1.0)[:,:] 
    obj_cube[:,:,2] = obj_rms[:,:]
    obj_cube[:,:,3] = obj_spatial[:,:]

    pysalt.clobberfile("data_preflat.fits")
    fits.PrimaryHDU(data=obj_cube[:,:,1]).writeto("data_preflat.fits", clobber=True)

    if (not type(skyline_flat) == type(None)):
        # We also received a skyline flatfield for field flattening
        obj_cube[:,:,1] /= skyline_flat.reshape((-1,1))
        logger.info("Applying skyline flatfield to data before sky-subtraction")
        #return 1,2
        pass

    pysalt.clobberfile("data_postflat.fits")
    fits.PrimaryHDU(data=obj_cube[:,:,1]).writeto("data_postflat.fits", clobber=True)

    
    # mask_objects = False
    if (not mask_objects and select_region == None):
        obj_bpm  = numpy.array(obj_hdulist['BPM'].data).flatten()
    else:
        
        use4sky = numpy.ones((obj_cube.shape[0]), dtype=numpy.bool)
        # by default, use entire frame for sky
 
        if (mask_objects):
            source_mask = find_source_mask(obj_data)
            use4sky = use4sky & (~source_mask)
            # trim down sky by regions not contaminated with (strong) sources

        if (not select_region == None):
            sky = numpy.zeros((obj_cube.shape[0]), dtype=numpy.bool)
            for y12 in select_region:
                print "@@@@@@@@@@",y12, numpy.sum(use4sky), use4sky.shape
                sky[y12[0]:y12[1]] = True
            use4sky = use4sky & sky
            # also only select regions explicitely chosen as sky

        print "selecting:", use4sky.shape, numpy.sum(use4sky)

        obj_cube = obj_cube[use4sky]

        _x = numpy.array(obj_data)
        _x[source_mask] = numpy.NaN
        pysalt.clobberfile("obj_mask.fits")
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=obj_data),
            fits.ImageHDU(data=_x)]).writeto("obj_mask.fits")

        obj_bpm  = numpy.array(obj_hdulist['BPM'].data)[use4sky].flatten()
        print obj_bpm.shape, obj_cube.shape


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

        print sky_regions
        logger.info("Selecting sky-pixels from user-defined regions")
        is_sky = numpy.zeros((obj_cube.shape[0]), dtype=numpy.bool)
        for idx, sky_region in enumerate(sky_regions):
            logger.debug("Good region: %d ... %d" % (sky_region[0], sky_region[1]))
            in_region = (obj_cube[:,3] > sky_region[0]) & \
                        (obj_cube[:,3] < sky_region[1]) & \
                        (numpy.isfinite(obj_cube[:,1]))
            is_sky[in_region] = True

        obj_cube = obj_cube[is_sky]
        
    allskies = obj_cube #[::skiplength]
    numpy.savetxt("xxx1", allskies)

    # _x = fits.ImageHDU(data=obj_hdulist['SCI.RAW'].data, 
    #                      header=obj_hdulist['SCI.RAW'].header)
    # _x.name = "STEP1"
    # obj_hdulist.append(_x)



    # #
    # # Load and prepare data
    # #
    # allskies = numpy.loadtxt(allskies_filename)

    # just to be on the safe side, sort allskies by wavelength
    sky_sort_wl = numpy.argsort(allskies[:,0])
    allskies = allskies[sky_sort_wl]
    numpy.savetxt("xxx2", allskies[::skiplength])

    logger.debug("Working on %7d data points" % (allskies.shape[0]))


    #
    # Compute cumulative distribution
    #
    logger.info("Computing cumulative distribution")
    allskies_cumulative = numpy.cumsum(allskies[:,1], axis=0)

    # print allskies.shape, allskies_cumulative.shape, wl_sorted.shape

    numpy.savetxt("cumulative.asc", 
                  numpy.append(allskies[::skiplength][:,0].reshape((-1,1)),
                               allskies_cumulative[::skiplength].reshape((-1,1)),
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
        bounds_error=False,
        fill_value=-9999,
        #assume_sorted=True,
        )

    logger.debug("Cumulative flux range: %f ... %f" % (
        allskies_cumulative[0], allskies_cumulative[1]))

    # now create the raw basepoints in cumulative flux space
    k_cumflux = numpy.linspace(allskies_cumulative[0],
                               allskies_cumulative[-1],
                               N_points+2)[1:-1]

    # and using the interpolator, convert flux space into wavelength
    k_wl = interp(k_cumflux)

    # eliminate all negative-wavelength basepoints - 
    # these represent interpolation errors
    k_wl = k_wl[k_wl>0]

    numpy.savetxt("opt_basepoints", 
                  numpy.append(k_wl.reshape((-1,1)),
                               k_cumflux.reshape((-1,1)),
                               axis=1)
                  )


    #############################################################################
    #
    # Add additional wavelength sampling points along the line-edges if
    # this was requested. 
    #
    #############################################################################
    if (add_edges):
        logger.info("Adding sky-samples for line edges")
        
        
        dl = 1.
        dn = 10

        if (use_fast_edges):
            edges = fastedge.find_line_edges(allskies, line_sigma=2.75)

            # distribute additional basepoints across 2. (+/- dl) angstroem 
            # for each edge
            all_edge_points = numpy.empty((edges.shape[0], dn))
            for ie, edge in enumerate(edges):
                bp = numpy.linspace(edge-dl, edge+dl, dn)
                all_edge_points[ie,:] = bp[:]

        else:
            pysalt.clobberfile("edges.cheat")
            if (not os.path.isfile("edges.cheat")):
                edges = find_edges_of_skylines.find_edges_of_skylines(allskies, fn="XXX")
                numpy.savetxt("edges.cheat", edges)
            else:
                edges = numpy.loadtxt("edges.cheat")

            # distribute additional basepoints across 2. (+/- dl) angstroem 
            # for each edge
            all_edge_points = numpy.empty((edges.shape[0], dn))
            for ie, edge in enumerate(edges[:,0]):
                bp = numpy.linspace(edge-dl, edge+dl, dn)
                all_edge_points[ie,:] = bp[:]
        
        # 
        # Now merge the list of new basepoints with the existing list.
        # sort this list ot make it a suitable input for spline fitting
        #
        numpy.savetxt("k_wl.in", k_wl)
        k_wl_new = numpy.append(k_wl, all_edge_points.flatten())
        k_wl = numpy.sort(k_wl_new)
        numpy.savetxt("k_wl.out", k_wl)

    #############################################################################
    #
    # Now we have the new optimal set of base points, let's compare it to the 
    # original with the same number of basepoints, sampling the available data
    # with points equi-distant in wavelength space.
    #
    #############################################################################


    wl_min, wl_max = numpy.min(allskies[:,0]), numpy.max(allskies[:,0])
    logger.debug("Min/Max WL: %.3f / %.3f" % (wl_min, wl_max))

    if (compare):
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
    logger.debug("#datapoints: %d, #basepoints: %d" % (
        allskies.shape[0], k_wl.shape[0]))

    k_opt_good = satisfy_schoenberg_whitney(allskies[:,0], k_wl, k=3)

    numpy.savetxt("allskies", allskies)
    fits.PrimaryHDU(data=allskies).writeto("allskies.fits", clobber=True)
    numpy.savetxt("bp_in", k_wl)
    numpy.savetxt("bp_out", k_opt_good)

    
    try:
        spline_opt = scipy.interpolate.LSQUnivariateSpline(
            x=allskies[:,0], 
            y=allskies[:,1], 
            t=k_opt_good[::10], #k_wl,
            w=None, # no weights (for now)
            bbox=[wl_min, wl_max], 
            k=3, # use a cubic spline fit
        )
    except ValueError:
        logger.error("Unable to compute LSQUnivariateSpline (data: %d, bp=%d/10)" % (
            allskies.shape[0], k_opt_good.shape[0]))
        return None, None

    spec_simple = numpy.append(k_wl.reshape((-1,1)),
                                             spline_opt(k_wl).reshape((-1,1)),
                                             axis=1)
    numpy.savetxt("spline_opt", spec_simple)

    #
    #
    # Now we have a pretty good guess on what the entire sky spectrum looks like
    # This means we can use known sky-lines to find and compensate for intensity 
    # variations
    #
    #
    if (not iterate):
        if (return_2d):

            pass
        else:
            # only return a 1-d spectrum, centered on the middle row 
            line = obj_wl.shape[0]/2
            wl = obj_wl[line,:]
            spec = spline_opt(wl)

            return spec


    # _x = fits.ImageHDU(data=obj_hdulist['SCI.RAW'].data, 
    #                      header=obj_hdulist['SCI.RAW'].header)
    # _x.name = "STEP2"
    # obj_hdulist.append(_x)


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

        try:
            spline_iter = scipy.interpolate.LSQUnivariateSpline(
                x=good_data[:,0], #allskies[:,0],#[good_point], 
                y=good_data[:,1], #allskies[:,1],#[good_point], 
                t=k_iter_good, #k_wl,
                w=None, # no weights (for now)
                bbox=[wl_min, wl_max], 
                k=3, # use a cubic spline fit
            )
        except ValueError:
            # this is most likely 
            # ValueError: Interior knots t must satisfy Schoenberg-Whitney conditions
            if (iteration > 0):
                break
            else:
                print "unable to compute LSQ spline, skipping 4/5 basepoints"
                spline_iter = scipy.interpolate.LSQUnivariateSpline(
                    x=good_data[:,0], #allskies[:,0],#[good_point], 
                    y=good_data[:,1], #allskies[:,1],#[good_point], 
                    t=k_iter_good[5:-5][::5], #k_wl,
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


    # _x = fits.ImageHDU(data=obj_hdulist['SCI.RAW'].data, 
    #                      header=obj_hdulist['SCI.RAW'].header)
    # _x.name = "STEP3"
    # obj_hdulist.append(_x)

    if (compare):
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
    sky2d = None

    # compute high-res sky-spectrum
    wl_highres = numpy.linspace(allskies[0,0], allskies[-1,0], 100000)
    sky_highres = spline_iter(wl_highres)
    numpy.savetxt("sky_highres", numpy.append(wl_highres.reshape((-1,1)),
                                              sky_highres.reshape((-1,1)), axis=1))

    allskies_synth = spline_iter(allskies[:,0])
    ascombined = numpy.zeros((allskies_synth.shape[0],4))
    ascombined[:,0] = allskies[:,0]
    ascombined[:,1] = allskies[:,1]
    ascombined[:,2] = allskies_synth[:]
    ascombined[:,3] = allskies[:,1] - allskies_synth[:]
    numpy.savetxt("skysub_all", ascombined)


    if (not spline_iter == None):
        padded = numpy.empty((obj_wl.shape[0], obj_wl.shape[1]+2))
        padded[:, 1:-1] = obj_wl[:,:]
        padded[:,0] = obj_wl[:,0]
        padded[:,-1] = obj_wl[:,-1]
        from_wl = 0.5*(padded[:, 0:-2] + padded[:, 1:-1])
        to_wl = 0.5*(padded[:, 1:-1] + padded[:, 2:])
        print "n\nXXXXX",from_wl.shape, to_wl.shape, obj_wl.shape, padded, "\nXXXXX\n"

        # this would be a nice call, but xxx.integral does not support multiple values
        # sky2d_x = spline_iter.integral(from_wl.ravel(), to_wl.ravel()).reshape(from_wl.shape)
        #
        # therefore we need to do the integration for each pixel by hand
        t0 = time.time()
        sky2d = numpy.array([spline_iter.integral(a,b) for a,b in zip(from_wl.ravel(),to_wl.ravel())]).reshape(obj_wl.shape)
        t1 = time.time()
        print "integration took %f seconds" % (t1-t0)
        fits.PrimaryHDU(data=sky2d).writeto("IntegSky.fits", clobber=True)

        # t0 = time.time()
        # sky2d = spline_iter(obj_wl.ravel()).reshape(obj_wl.shape)
        # t1 = time.time()
        # print "interpolation took %f seconds" % (t1-t0)

        #if (not type(skyline_flat) == type(None)):
        #    sky2d *= skyline_flat.reshape((-1,1))

        # skysub = obj_data - sky2d
        # ss_hdu = fits.ImageHDU(header=obj_hdulist['SCI.RAW'].header,
        #                          data=skysub)
        # ss_hdu.name = "SKYSUB.OPT"
        # obj_hdulist.append(ss_hdu)

        # ss_hdu2 = fits.ImageHDU(header=obj_hdulist['SCI.RAW'].header,
        #                          data=sky2d)
        # ss_hdu2.name = "SKYSUB.IMG"
        # obj_hdulist.append(ss_hdu2)


        return sky2d, spline_iter, (x_eff, wl_map, medians, p_scale, p_skew, fm)




def estimate_slit_intensity_variations(hdulist, spline, sky_2d):

    # We can use a horizontal cut through the 2-D sky as a template of the 
    # sky-spectrum to extract sky emission lines
    row = int(sky_2d.shape[0]/2)
    skyspec = sky_2d[row,:]

    skyline_list = wlcal.find_list_of_lines(skyspec, readnoise=1, avg_width=1)
    print skyline_list
    numpy.savetxt("skyline_list", skyline_list)

    #
    # Select a couple of the strong lines
    #



    #
    # Now trace the arclines and do the subpixel centering so we can compute 
    # the line intensity profiles
    #

    # data = hdulist['RAW'].data
    
    #subpixel_centroid_trace(data, tracedata, width=5, dumpfile=None)



if __name__ == "__main__":


    logger_setup = pysalt.mp_logging.setup_logging()


    obj_fitsfile = sys.argv[1]
    obj_hdulist = fits.open(obj_fitsfile)

    sky_regions = None
    if (len(sys.argv) > 2):
        user_sky = sys.argv[2]
        sky_regions = numpy.array([x.split(":") for x in user_sky.split(",")]).astype(numpy.int)

    # obj_mask = find_source_mask(obj_hdulist['SCI.RAW'].data)

    simple_spec = optimal_sky_subtraction(obj_hdulist, 
                                          sky_regions=sky_regions,
                                          N_points=1000,
                                          iterate=False,
                                          skiplength=10, 
                                          compare=True,
                                          mask_objects=True,
                                          return_2d=False)
    numpy.savetxt("simple_spec", simple_spec)


    skyline_list = wlcal.find_list_of_lines(simple_spec, readnoise=1, avg_width=1)
    print skyline_list

    i, ia, im = skyline_intensity.find_skyline_profiles(obj_hdulist, skyline_list)
    
    #numpy.savetxt("skyline_list", skyline_list)


    sky_2d, spline = optimal_sky_subtraction(obj_hdulist, 
                                             sky_regions=sky_regions,
                                             N_points=1000,
                                             iterate=False,
                                             skiplength=5,
                                             mask_objects=True,
                                             skyline_flat=ia)

    # # Now use the spline interpolator to create a list of strong skylines
    # estimate_slit_intensity_variations(obj_hdulist, spline, sky_2d)

    obj_hdulist.writeto(obj_fitsfile[:-5]+".optimized.fits", clobber=True)

    pysalt.mp_logging.shutdown_logging(logger_setup)
