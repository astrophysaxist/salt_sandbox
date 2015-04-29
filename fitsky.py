#!/usr/bin/env python

import numpy, scipy, scipy.spatial, scipy.interpolate, os, sys
import matplotlib.pyplot as pl


if __name__ == "__main__":
    data = numpy.load(sys.argv[1])
    
    x_range = [int(x) for x in sys.argv[2].split(":")]
    x = data[:,0] #numpy.load("spline_x.npy")
    y = data[:,1] #numpy.load("spline_y.npy")
    t = numpy.loadtxt("spline_t")

    wl_min, wl_max = numpy.min(x), numpy.max(x)

    # Now reject all basepoints with insufficient datapoints close to them
    # require at least N datapoints
    kdtree = scipy.spatial.cKDTree(x.reshape((-1,1)))
    N_min = 10
    search_radius = t[1] - t[0]
    nearest_neighbor, i = kdtree.query(x=t.reshape((-1,1)), 
                                       k=N_min, # only find 1 nearest neighbor
                                       p=1, # use linear distance
                                       distance_upper_bound=search_radius)
    neighbor_count = numpy.sum( numpy.isfinite(nearest_neighbor), axis=1)
    print neighbor_count.shape
    
    numpy.savetxt("neighbor_count", 
                  numpy.append(t.reshape((-1,1)),
                               neighbor_count.reshape((-1,1)), axis=1)
                  )

    #
    # Now eliminate all basepoints with not enough data points for proper fitting
    #
    t = t[neighbor_count >= N_min]

    sky_spectrum_spline = scipy.interpolate.LSQUnivariateSpline(
                x, y, t,
                w=None, # no weights (for now)
                bbox=[wl_min, wl_max], 
                k=3, # use a cubic spline fit
                )

    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(x_range) #(5850,5950))
    ax.set_ylim((0,3800))
    
    ax.scatter(x,y, linewidths=0) #,s=1,marker=",")
    ax.scatter(t,numpy.ones_like(t)*400, linewidths=0, c='r')
    ax.plot(t, sky_spectrum_spline(t), 'g-', linewidth=2)

    fig.show()
    pl.show()

    #numpy.savetxt("spline_xy",
    #              numpy.append(x.reshape((-1,1)),
    #                           y.reshape((-1,1)), axis=1)[::50])

    print "\n\nknots"
    print sky_spectrum_spline.get_knots().shape
    print sky_spectrum_spline.get_knots()

    ss = numpy.append(t.reshape((-1,1)),
                      sky_spectrum_spline(t).reshape((-1,1)),
                      axis=1)
    numpy.savetxt("skyspectrum.txt", ss)
