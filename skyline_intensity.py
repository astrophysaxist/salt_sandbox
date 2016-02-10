#!/usr/bin/env python

import os, sys, pyfits, numpy
import bottleneck
import logging

from wlcal import lineinfo_colidx
import traceline
import scipy, scipy.stats



def compute_local_median_std(tracedata, intensity, window=5):

    med_std = numpy.zeros((tracedata.shape[0],2))

    for idx, y in enumerate(tracedata[:,0]):
        # print y
        
        # select all rows within the given window 
        sel_y = (numpy.fabs(tracedata[:,0] - y) <= window) & numpy.isfinite(intensity)
        if (numpy.sum(sel_y) > window):
            # we have enough data points for a proper median computation

            sigma_med = scipy.stats.scoreatpercentile(intensity[sel_y], [50,16,84])
            med_std[idx,0] = sigma_med[0]
            med_std[idx,1] = (sigma_med[2] - sigma_med[1])/2.

            #med_std[idx,0] = bottleneck.nanmedian(intensity[sel_y])
            #med_std[idx,1] = bottleneck.nanstd(intensity[sel_y])

    return med_std


def find_skyline_profiles(hdulist, skyline_list, data=None, 
                          write_debug_data=False, 
                          tracewidth=10,
                          use_coords='pixel',
                          line=-1,
                          n_lines_max=15,
                          min_signal_to_noise=10,
                          max_intensity=25000,
):


    logger = logging.getLogger("SkylineProfiles")

    if (data == None):
        data = hdulist['SCI.RAW'].data

    if (line < 0):
        line = int(data.shape[0]/2)

    if (use_coords == 'pixel'):
        pass
    else:
        pass

    wl = hdulist['WAVELENGTH'].data

    line_wl = wl[line,:]

    print line_wl

    #
    # Pick lines that are separated from each other to avoid confusion
    #
    good_lines = traceline.pick_line_every_separation(
        skyline_list,
        trace_every=5,
        min_line_separation=40,
        n_pixels=wl.shape[1],
        min_signal_to_noise=10,
        )
    skyline_list = skyline_list[good_lines]

    s2n_sort = numpy.argsort(skyline_list[:,4])[::-1]
    skyline_list = skyline_list[s2n_sort]

    n_lines = numpy.min(numpy.array([skyline_list.shape[0], n_lines_max]))
    profiles = numpy.empty((data.shape[0], n_lines))

    min_intensity = 600

    for idx in range(n_lines):
        
        #
        # Make sure we do not include lines with close neighbors here,
        # or at least combine these two lines into one to get an average 
        # profile - otherwise we likely run into problems with our simplistic 
        # way of estimating backgrounds by interpolating linearly between the 
        # left and right edge of the window around each line
        #

        #
        # Select the closest pixel in wavelength to the wavelength 
        # of the skyline
        #

        # first, convert the pixelposition of the line into a proper wavelength
        pixelpos = skyline_list[idx, lineinfo_colidx['PIXELPOS']]
        line_wavelength = wl[line, int(pixelpos)]

        s2n = skyline_list[idx, lineinfo_colidx['S2N']]
        logger.info("Tracing line: X=%d, L=%.3f, S/N=%.2f" % (pixelpos, line_wavelength, s2n))
        # print skyline_list[idx, lineinfo_colidx['WAVELENGTH']],\
        #     skyline_list[idx, lineinfo_colidx['PIXELPOS']],\
        #     line_wavelength, \
        #     skyline_list[idx, lineinfo_colidx['S2N']]

        # then use the wavelength map to track the line across the focalplane
        closest = numpy.argmin(wl < line_wavelength, axis=1)
        # print closest

        # Now we should have data for a nice trace 
        tracedata = numpy.append(
            numpy.arange(data.shape[0]).reshape((-1,1)),
            closest.reshape((-1,1)), 
            axis=1)

        pos, intensity, linedata, bgsubdata = traceline.subpixel_centroid_trace(
            data, tracedata, width=tracewidth, 
            dumpfile="skyline_%02d.fits" % (idx+1) if write_debug_data else None,
            return_all=True)

        raw_flux = bottleneck.nansum(linedata, axis=1)
        
        #
        # Now compute the local standard deviation around each pixel
        #
        medstd = compute_local_median_std(tracedata, intensity)

        if (write_debug_data):
            fn = "sky_trace.%d" % (idx+1)
            numpy.savetxt(fn,
                          numpy.append(tracedata,
                                       intensity.reshape((-1,1)), axis=1))
            logger.info("Writing tracedata to %s" % (fn))

        combined = numpy.empty((tracedata.shape[0], tracedata.shape[1]+5))
        combined[:,:tracedata.shape[1]] = tracedata[:]
        combined[:,tracedata.shape[1]] = intensity[:]
        combined[:,tracedata.shape[1]+1] = raw_flux
        combined[:,tracedata.shape[1]+2] = bottleneck.nanmax(linedata, axis=1)
        combined[:,-2:] = medstd[:]
        
        numpy.savetxt("sky_trace_comb.%d" % (idx+1), combined)    

        #
        # Now reject all intensities that exceed the local variance by 
        # more than 3 sigma
        #
        outlier = (intensity > medstd[:,0]+3*medstd[:,1]) | \
                  (intensity < medstd[:,0]-3*medstd[:,1])
        intensity[outlier] = numpy.NaN

        profiles[:,idx] = intensity[:]

    #
    # Now we have all profiles, compute median and noise stats
    #
    central_portion = profiles[int(0.1*profiles.shape[0]):int(0.9*profiles.shape[0]), :]

    # compute median intensity near the center of the slit
    central_median = bottleneck.nanmedian(central_portion, axis=0).reshape((1,-1))
    print central_median.shape

    # Reject all skylines below a certain intensity threshold
    bright_enough = central_median[0,:] > min_intensity
    #
    print bright_enough
    print "profiles all:", profiles.shape
    bright_lines = numpy.arange(profiles.shape[1])[bright_enough]
    print bright_lines
    profiles = profiles[:,bright_lines]
    print "profiles bright:", profiles.shape
    
    central_portion = central_portion[:,bright_lines]
    central_median = central_median[:,bright_lines]


    norm_profiles = profiles / central_median
    norm_noise = bottleneck.nanstd(central_portion/central_median, axis=0)

    print central_median
    print norm_noise

    #
    # Now compute the weighted average of all lines
    #
    weights = numpy.ones_like(profiles)
    weights[numpy.isnan(profiles)] = 0.
    weights /= norm_noise

    weighted_avg = bottleneck.nansum(norm_profiles*weights, axis=1) / numpy.sum(weights, axis=1)
    print weighted_avg.shape

    if (write_debug_data):
        numpy.savetxt("sky_trace.wavg",
                      numpy.append(numpy.arange(weighted_avg.shape[0]).reshape((-1,1)),
                                   weighted_avg.reshape((-1,1)), axis=1))
        
    #
    # Now in a final step, average/median filter the results to create a higher
    # signal-to-noise correction
    #
    w = 15
    # add some padding on either end to avoid running into no-man's memory land
    padded = numpy.empty((weighted_avg.shape[0]+2*w))
    padded[w:-w] = weighted_avg[:]
    padded[:w] = numpy.NaN
    padded[-w:] = numpy.NaN

    blkavg = numpy.array(
        [bottleneck.nanmean(padded[i-w:i+w+1]) for i in range(w,padded.shape[0]-w)])
    blkmedian = numpy.array(
        [bottleneck.nanmedian(padded[i-w:i+w+1]) for i in range(w,padded.shape[0]-w)])
    #weighted_avg.shape[0])])

    if (write_debug_data):
        numpy.savetxt("sky_trace.wavg_avg",
                      numpy.append(numpy.arange(weighted_avg.shape[0]).reshape((-1,1)),
                                   blkavg.reshape((-1,1)), axis=1))
        numpy.savetxt("sky_trace.wavg_med",
                      numpy.append(numpy.arange(weighted_avg.shape[0]).reshape((-1,1)),
                                   blkmedian.reshape((-1,1)), axis=1))

    #
    # Now return the raw intensity profile, and the median and average 
    # filtered ones as well
    #
    return weighted_avg, blkavg, blkmedian

        # # normalize intensity 
        # if (numpy.isnan(central_median)):
        #     print "this line is useless"
        #     continue

        # norm_intensity = intensity / central_median
        # numpy.savetxt("sky_trace1.%d" % (idx+1),
        #               numpy.append(tracedata,
        #                            norm_intensity.reshape((-1,1)), axis=1))
        # norm_noise = bottleneck.nanstd(central_slit/central_median)

        # # Now we have a nice line profile, so smooth it to remove potential 
        # # outlier pixels
        # w=25
        # median_smoothed = numpy.array(
        #     [bottleneck.nanmedian(intensity[i-w:i+w+1]) 
        #      for i in range(intensity.shape[0])])

        # numpy.savetxt("sky_trace2.%d" % (idx+1),
        #               numpy.append(tracedata,
        #                            median_smoothed.reshape((-1,1)), axis=1))
        # print idx+1, norm_noise


    


if __name__ == "__main__":

    skyline_list = numpy.loadtxt("skyline_list")

    fitsfile = sys.argv[1]
    hdulist = pyfits.open(fitsfile)

    
    i, i_med, i_avg = find_skyline_profiles(hdulist, skyline_list)
