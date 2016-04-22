#!/usr/bin/env python


import os, sys, numpy, scipy.ndimage, scipy.signal
import logging
from astropy.io import fits

import pysalt.mp_logging


def find_edges_of_skylines(allskies, fn):

    logger = logging.getLogger("FindSkyEdges")

    logger.debug("median filter")
    #order0 = scipy.ndimage.filters.median_filter(allskies[:,1].reshape((-1,1)), size=49, mode='mirror')[:,0]
    order0 = scipy.ndimage.filters.gaussian_filter(input=allskies[:,1], order=0, sigma=15)
    numpy.savetxt(fn+".diff0", 
                  numpy.append(allskies[:,0].reshape((-1,1)),
                               order0.reshape((-1,1)), axis=1))

    logger.debug("1st order")
    order1 = numpy.diff(order0)
    numpy.savetxt(fn+".diff1", 
                  numpy.append(allskies[:-1,0].reshape((-1,1)),
                               order1.reshape((-1,1)), axis=1))
    #order1s = scipy.ndimage.filters.gaussian_filter(input=order1, order=0, sigma=3)
    order1s = scipy.ndimage.filters.median_filter(order1.reshape((-1,1)), size=15, mode='mirror')[:,0]
    numpy.savetxt(fn+".diff1s", 
                  numpy.append(allskies[:-1,0].reshape((-1,1)),
                               order1s.reshape((-1,1)), axis=1))


    # print "2nd order"
    # order2 = numpy.diff(order1)
    # numpy.savetxt(fn+".diff2", 
    #               numpy.append(allskies[:-2,0].reshape((-1,1)),
    #                            order2.reshape((-1,1)), axis=1))


    logger.info("1st order noise")

    good = numpy.isfinite(order1)
    for i in range(3):
        _med = numpy.median(order1[good])
        _std = numpy.std(order1[good])
        print _med, _std
        good = (order1 > _med-3*_std) & (order1 < _med+3*_std)
        print numpy.sum(good)
        #numpy.savetxt("obj_mask.filter%d" % (i+1), combined[good])

    logger.info("results for 1st order noise: %f +/- %f"  %(_med, _std))


    # 
    #  Now find edges
    #
    logger.info("finding edges")
    _edges = scipy.signal.find_peaks_cwt(
        vector=numpy.fabs(order1), 
        widths=numpy.array([10,20]), 
        wavelet=None, 
        max_distances=None, gap_thresh=None, min_length=None, min_snr=1, noise_perc=10)
    # print _edges

    wl_edges = allskies[:,0][_edges]
    # print wl_edges
    numpy.savetxt("edges_all", wl_edges)

    s2n = numpy.fabs(order1[_edges]) / _std
    s2n_cutoff = 3.
    is_strong_edge = (s2n > s2n_cutoff)
    strong_edges = allskies[:,0][_edges][is_strong_edge]

    numpy.savetxt("edges_s2n=3", wl_edges)

    combined = numpy.append(strong_edges.reshape((-1,1)),
                               (s2n[is_strong_edge]).reshape((-1,1)), axis=1)

    numpy.savetxt("edges_s2n=3.x", combined
                  )

    #source = ~good
    logger.info("done, returing list of edges")
    return combined



if __name__ == "__main__":

    logger_setup = pysalt.mp_logging.setup_logging()

    fn = sys.argv[1]

    allskies = numpy.loadtxt(fn)
    print allskies.shape


    edges = find_edges_of_skylines(allskies, fn)

    print edges
    pysalt.mp_logging.shutdown_logging(logger_setup)
