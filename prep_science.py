#!/usr/bin/env python

import warnings

import os, sys, scipy, scipy.stats, scipy.ndimage, pyfits, numpy
import wlcal
import skyline_intensity
import traceline

from optimal_spline_basepoints import satisfy_schoenberg_whitney
import bottleneck
import logging

warnings.simplefilter('ignore', UserWarning)

from wlcal import lineinfo_colidx

import pysalt.mp_logging

import matplotlib.pyplot as pyplot

def compute_smoothed_profile(data_x, data_y, 
                             n_iterations=3,
                             n_max_neighbors=100, # pixels
                             avg_sample_width=50, # pixels
                             n_sigma=3,
):

    logger = logging.getLogger("FitSplineToNoisyData")

    valid = numpy.isfinite(data_x) & numpy.isfinite(data_y)
    if (numpy.sum(valid) < data_x.shape[0]):
        # there is at least one NaN that we need to replace
        
        #data_x = data_x[valid]
        data_y[~valid] = -1e9 # = data_y[valid]

    wl_min, wl_max = numpy.min(data_x), numpy.max(data_x)
    #avg_sample_width = (wl_max - wl_min) / k_basepoints.shape[0] * 2

    weights = numpy.ones(data_x.shape)


    for iteration in range(n_iterations):

        # Create a KD-tree with all data points
        wl_tree = scipy.spatial.cKDTree(data_x.reshape((-1,1)))

        # Now search this tree for points near each of the spline base points
        d, i = wl_tree.query(data_x.reshape((-1,1)),
                             k=100, # use 100 neighbors
                             distance_upper_bound=avg_sample_width)

        #
        # Compute standard deviation around every pixel
        #
        bad = (i >= data_x.shape[0])
        i[bad] = 0

        # Now we have all indices of a bunch of nearby datapoints, so we can 
        # extract how far off each of the data points is
        neighbors = data_y[i] #dflux[i]
        neighbors[bad] = numpy.NaN
        #print "dflux_2d = ", delta_flux_2d.shape

        # With this we can estimate the scatter around each spline fit basepoint
        local_var = bottleneck.nanstd(neighbors, axis=1)
        # print "variance:", local_var.shape
        numpy.savetxt("fit_variance.iter_%d" % (iteration+1),
                      numpy.append(data_x.reshape((-1,1)),
                                   local_var.reshape((-1,1)), axis=1))

        local_median = bottleneck.nanmedian(neighbors, axis=1)
        numpy.savetxt("fit_median.iter_%d" % (iteration+1),
                      numpy.append(data_x.reshape((-1,1)),
                                   local_median.reshape((-1,1)), axis=1))

        #
        # Now reject all pixels outside the 2-3 sigma range of the local scatter
        #
        outliers = (data_y > local_median+n_sigma*local_var) | \
                   (data_y < local_median-n_sigma*local_var)
        data_y[outliers] = numpy.NaN

        # # compute spline
        # k_iter_good = satisfy_schoenberg_whitney(data_x, k_basepoints, k=3)

        # spline_iter = scipy.interpolate.LSQUnivariateSpline(
        #     x=data_x, #allskies[:,0],#[good_point], 
        #     y=data_y, #allskies[:,1],#[good_point], 
        #     t=k_iter_good, #k_basepoints,
        #     w=None, #weights, #None, # no weights (for now)
        #     bbox=[wl_min, wl_max], 
        #     k=3, # use a cubic spline fit
        #     )
        # numpy.savetxt("spline_opt.iter%d" % (iteration+1), 
        #               numpy.append(k_basepoints.reshape((-1,1)),
        #                            spline_iter(k_basepoints).reshape((-1,1)),
        #                            axis=1)
        #           )

        # # compute spline fit for each wavelength data point
        # dflux = data_y - spline_iter(data_x)
        # print dflux

        # numpy.savetxt("dflux_%d" % (iteration),
        #               numpy.append(data_x.reshape((-1,1)),
        #                            dflux.reshape((-1,1)), axis=1),
        # )



        # # #
        # # # Add here: work out the scatter of the distribution of pixels in the 
        # # #           vicinity of this basepoint. This is what determined outlier 
        # # #           or not, and NOT the uncertainty in a given pixel
        # # #


        # # make sure to flag outliers
        # bad = (i >= dflux.shape[0])
        # i[bad] = 0

        # # Now we have all indices of a bunch of nearby datapoints, so we can 
        # # extract how far off each of the data points is
        # delta_flux_2d = data_x[i] #dflux[i]
        # delta_flux_2d[bad] = numpy.NaN
        # print "dflux_2d = ", delta_flux_2d.shape

        # # With this we can estimate the scatter around each spline fit basepoint
        # var = bottleneck.nanstd(delta_flux_2d, axis=1)
        # print "variance:", var.shape
        # numpy.savetxt("fit_variance.iter_%d" % (iteration+1),
        #               numpy.append(k_basepoints.reshape((-1,1)),
        #                            var.reshape((-1,1)), axis=1))

        # #
        # # Now interpolate this scatter linearly to the position of each 
        # # datapoint in the original dataset. That way we can easily decide, 
        # # for each individual pixel, if that pixel is to be considered an 
        # # outlier or not.
        # #
        # # Note: Need to consider ALL pixels here, not just the good ones 
        # #       selected above
        # #
        # std_interpol = scipy.interpolate.interp1d(
        #     x = k_basepoints, 
        #     y = var,
        #     kind = 'linear',
        #     fill_value=1e3,
        #     bounds_error=False,
        #     #assume_sorted=True
        #     )
        # var_at_pixel = std_interpol(data_x)

        # numpy.savetxt("pixelvar.%d" % (iteration+1), 
        #               numpy.append(data_x.reshape((-1,1)),
        #                            var_at_pixel.reshape((-1,1)), axis=1))

        # # Now mark all pixels exceeding the noise threshold as outliers
        # not_outlier = numpy.fabs(dflux) < var_at_pixel

        # # good_data = good_data[not_outlier]
        # # numpy.savetxt("good_after.%d" % (iteration+1),
        # #               numpy.append(data_x.reshape((-1,1)),
        # #                            dflux[not_outlier].reshape((-1,1)), axis=1))

        # # logger.info("Done with iteration %d (%d pixels left)" % (iteration+1, good_data.shape[0]))

    return local_median


def filter_isolate_skylines(data, write_debug_data=False):

    logger = logging.getLogger("FilterIsolateSkylines")

    #
    # run a median filter in x-direction to eliminate continua
    #
    logger.info("Filtering to separate continuum and skylines")
    data_filtered = scipy.ndimage.filters.median_filter(
        input=data, 
        size=(1,75), 
        footprint=None, 
        output=None, 
        mode='reflect', 
        cval=0.0, 
        origin=0)


    skylines = data - data_filtered

    if (write_debug_data):
        logger.debug("Writing debug information to FITS")
        pyfits.PrimaryHDU(data=data).writeto("data.fits", clobber=True)
        pyfits.PrimaryHDU(data=data_filtered).writeto("filtered.fits", clobber=True)
        pyfits.PrimaryHDU(data=skylines).writeto("skylines.fits", clobber=True)

        fake_hdu = pyfits.HDUList([
            pyfits.PrimaryHDU(),
            pyfits.ImageHDU(data=skylines, name="SCI"),
        ])

    return skylines, data_filtered

def exclude_lines_close_to_chipgaps(
        chipgaps, 
        lines, 
        zoe=50, # zone of exclusion
        ):

    pos_x = lines[:,lineinfo_colidx['PIXELPOS']]

    close_to_gap = numpy.zeros(lines.shape[0], dtype=numpy.bool)
    # print close_to_gap

    # print 

    for idx, gap in enumerate(chipgaps):#range(gaplist.shape[0]):
        #print gap
        close_to_this_gap = (pos_x > (gap[0] - zoe)) & \
                            (pos_x < (gap[1] + zoe))
        #print close_to_this_gap

        close_to_gap = close_to_gap | close_to_this_gap

        #print

    #print pos_x[close_to_gap]

    return lines[~close_to_gap]





def extract_skyline_intensity_profile(
        hdulist, data, wls=None, 
        plot_filename=None,
        use_as_trace_data='skylines',
        write_debug_data=False):

    logger = logging.getLogger("NightskyFlats")

    skylines, continuum = filter_isolate_skylines(data, write_debug_data=write_debug_data)

    #
    # Now find lines
    #
    logger.info("Searching for night-sky lines")
    #nightsky_spec_1d = numpy.average(skylines[600:620,:], axis=0)
    nightsky_spec_1d = numpy.average(data[600:620,:], axis=0)
    #print nightsky_spec_1d.shape
    numpy.savetxt("nightsky", nightsky_spec_1d)
    #wlcal.extract_arc_spectrum(fake_hdu, line=600,avg_width=30)
    
    
    lines = wlcal.find_list_of_lines(nightsky_spec_1d, readnoise=2, avg_width=20,
                                     pre_smooth=2)
    print lines

    #
    # Eliminate all lines close to (within 50 pixels) any of the chip gaps
    #
    gapspec, chipgaps = traceline.find_chip_gaps(hdulist)
    lines = exclude_lines_close_to_chipgaps(
        chipgaps=chipgaps,
        lines=lines,
        zoe=50,
        )

    if (not wls == None):
        # We have a valid wavelength calibration, so convert pixel 
        # coordinates to wavelengths
        lines[:,lineinfo_colidx['WAVELENGTH']] = \
            numpy.polynomial.polynomial.polyval(
                lines[:,lineinfo_colidx['PIXELPOS']], 
                wls
            )

    #print lines
    numpy.savetxt("nightsky_lines", lines)

    #
    # Trace their intensity, and ompute a mean line intensity profile
    #

    #
    logger.info("Tracing emission line intensity profile")

    tracedata = data
    if (use_as_trace_data == 'skylines'):
        tracedata = skylines
    weighted_avg, blkavg, blkmedian = \
        skyline_intensity.find_skyline_profiles(
            hdulist, 
            lines, 
            line=610,
            data=tracedata,
            write_debug_data=write_debug_data,
            tracewidth=25,
            n_lines_max=5,
        )

    numpy.savetxt("skylines.weightedavg", weighted_avg)
    numpy.savetxt("skylines.blkavg", blkavg)
    numpy.savetxt("skylines.blkmedian", blkmedian)

    # weighted_avg = numpy.loadtxt("skylines.weightedavg")
    # blkavg = numpy.loadtxt("skylines.blkavg")
    # blkmedian = numpy.loadtxt("skylines.blkmedian")

    # Now fit a smoothing spline, while rejecting outliers
    n_points = 100
    k_basepoints = numpy.linspace(1, weighted_avg.shape[0]-2, 100)
    # print k_basepoints.shape[0]
    # print k_basepoints
    

    data_x = numpy.arange(weighted_avg.shape[0])
    intensity_profile = compute_smoothed_profile(data_x=data_x, 
                                                 data_y=weighted_avg, 
                                                 n_iterations=3,
                                            )
    numpy.savetxt("intensity_profile", intensity_profile)

    # Make sure we deal with very small and/or negative numbers appropriately
    intensity_profile[intensity_profile < 1e-3] = 1e-3
    numpy.savetxt("intensity_profile_v2", intensity_profile)

    #
    # Now we have the full-resolution skyline intensity profile,
    # use it to correct the data
    #
    numpy.savetxt("intensity_profile", intensity_profile)
    flat_skylines = skylines / intensity_profile.reshape((-1,1))
    pyfits.PrimaryHDU(data=flat_skylines).writeto("flat_skylines.fits", clobber=True)

    pyfits.PrimaryHDU(
        data=data/intensity_profile.reshape((-1,1))).writeto(
            "flat_data.fits", clobber=True)

    if (not plot_filename == None):
        logger.debug("Creating diagnostic plot for the line intensity profile")
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        ax.plot(weighted_avg, "-", c='blue')
        ax.plot(intensity_profile, "-", c='white', linewidth=3)
        ax.plot(intensity_profile, "-", c='black', linewidth=1)
        ax.set_xlabel("Position along slit")
        ax.set_ylabel("Relative intensity")
        ax.set_xlim((-0.03*weighted_avg.shape[0], 1.03*weighted_avg.shape[0]))
        fig.savefig(plot_filename)


    #
    # Return results
    #
    return skylines, lines, intensity_profile

    

#if (False):
if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()

    filename = sys.argv[1]
    hdulist = pyfits.open(filename)

    n_params = hdulist[0].header['WLSFIT_N']
    wls_fit = numpy.zeros(n_params)
    for i in range(n_params):
        wls_fit[i] = hdulist[0].header['WLSFIT_%d' % (i)]

    for iteration in range(3):

        data = hdulist['SCI.RAW'].data

        pyfits.PrimaryHDU(data=hdulist['SCI.RAW'].data).writeto(
            "skyflattened_pre%d.fits" % (iteration+1), clobber=True)

        skylines, lines, profile = extract_skyline_intensity_profile(
            hdulist, data, wls=wls_fit, write_debug_data=True,
            use_as_trace_data='skylines',
            plot_filename="slitprofile_%d.png" % (iteration+1))

        for i in range(10):
            src = "skyline_%02d.fits" % (i+1)
            dst = "skyline_%02d_iter%d.fits" % (i+1, iteration+1)
            pysalt.clobberfile(dst)
            try:
                os.rename(src, dst)
            except:
                pass


        for i in range(10):
            src = "sky_trace_comb.%d" % (i+1)
            dst = "sky_trace_comb_iter%d.%d" % (iteration+1, i+1)
            pysalt.clobberfile(dst)
            try:
                os.rename(src, dst)
            except:
                pass

        for i in range(10):
            src = "sky_trace.%d" % (i+1)
            dst = "sky_trace_iter%d.%d" % (iteration+1, i+1)
            pysalt.clobberfile(dst)
            try:
                os.rename(src, dst)
            except:
                pass

        try:
            pysalt.clobberfile("sky_trace_iter%d.wavg" % (iteration+1))
            os.rename("sky_trace.wavg", "sky_trace_iter%d.wavg" % (iteration+1))
        except:
            pass

         # for i in range(2):
         #    fn = "skyline_%02d_iter%d.fits" % (i+1, iteration+1)
         #    h = pyfits.open(fn)
         #    h[1].data /= profile.reshape((-1,1))
         #    h.writeto(fn[:-5]+".flat.fits", clobber=True)

        hdulist['SCI.RAW'].data /= profile.reshape((-1,1))

        numpy.savetxt("skyflat_%d.txt" % (iteration+1), profile.reshape((-1,1)))

        pyfits.PrimaryHDU(data=hdulist['SCI.RAW'].data).writeto(
            "skyflattened_%d.fits" % (iteration+1), clobber=True)

        break

    pysalt.mp_logging.shutdown_logging(logger_setup)
