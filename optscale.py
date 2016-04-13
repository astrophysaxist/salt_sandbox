#!/usr/bin/env python

import os, sys, pyfits
import numpy
import scipy, scipy.ndimage, scipy.optimize
import scipy.interpolate
import math


def scaled_sky(p, skyslice):
    return skyslice * p

def sky_residuals(p, imgslice, skyslice):
    ss = scaled_sky(p, skyslice)
    res = (imgslice - ss) #* skyslice
    return res[numpy.isfinite(res)]

if __name__ == "__main__":

    img_fn = sys.argv[1]
    img_hdu = pyfits.open(img_fn)

    sky_fn = sys.argv[2]
    sky_hdu = pyfits.open(sky_fn)

    img = img_hdu[0].data

    sky = sky_hdu[0].data

    print img.shape, sky.shape

    vert_size = 5
    n_slices = int(math.ceil(img.shape[0] / float(vert_size)))
    print n_slices
    
    scaling_data = numpy.zeros((n_slices,3))

    for curslice in range(n_slices):
        
        y0 = curslice * vert_size
        y1 = y0+vert_size if ( y0+vert_size < img.shape[0]) else img.shape[0]
        print y0,y1

        img_slice = img[y0:y1, :]
        sky_slice = sky[y0:y1, :]

        p_init = [1.0]

        fit_args = (img_slice, sky_slice)
        _fit = scipy.optimize.leastsq(
            sky_residuals,
            p_init, 
            args=fit_args,
            maxfev=500,
            full_output=1)
        #print _fit[0]

        # img_slice -= scaled_sky(_fit[0],sky_slice)

        scaling_data[curslice, :2] = [0.5*(y0+y1), _fit[0]]


    #
    # Now fit a low-order spline to the scaling profile
    #
    medfilt = scipy.ndimage.filters.median_filter(
        scaling_data[:,1],
        20,
        mode='wrap',
        )
    scaling_data[:,2] = medfilt[:]
    print medfilt

    interp = scipy.interpolate.InterpolatedUnivariateSpline(
        x=scaling_data[:,0],
        y=scaling_data[:,2],
        k=3,
#        bounds_error=False,
#        fill_value=0,
        )

    full_profile = interp(numpy.arange(img.shape[0]))
    numpy.savetxt("optscale.full", numpy.append(
        numpy.arange(full_profile.shape[0]).reshape((-1,1)),
        full_profile.reshape((-1,1)), axis=1))

    img -= sky*full_profile.reshape((-1,1))
    pyfits.PrimaryHDU(data=img).writeto(sys.argv[3], clobber=True)

    numpy.savetxt("optscale.out", scaling_data)
