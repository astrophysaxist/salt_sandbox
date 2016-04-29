#!/usr/bin/env python

import os, sys, pyfits
import numpy
import scipy, scipy.ndimage, scipy.optimize
import scipy.interpolate
import itertools
import math
import logging
import bottleneck


def scaled_sky(p, skyslice):
    return skyslice * p[0]

def sky_residuals(p, imgslice, skyslice):
    ss = scaled_sky(p, skyslice)
    _,x = numpy.indices(imgslice.shape)
    cont = p[2] #p[1]*x+p[2]
    res = (imgslice - (ss+cont)) #* skyslice
    return res[numpy.isfinite(res)]

def minimize_sky_residuals2(img, sky, wl, bpm, vert_size=5, smooth=3, debug_out=True, dl=-10):

    logger = logging.getLogger("SkyScaling2")

    # find block size in wavelength and spatial direction
    min_wl = bottleneck.nanmin(wl)
    max_wl = bottleneck.nanmax(wl)
    if (dl < 0):
        dl = (max_wl-min_wl)/numpy.fabs(dl)
    if (vert_size<0):
        vert_size = img.shape[0]/numpy.fabs(vert_size)

    logger.info("Using blocks of %d pixels and %d angstroems" % (
        vert_size, int(dl)))

    img = numpy.array(img)
    img[bpm > 0] = numpy.NaN

    n_wl_blocks = int(math.ceil((max_wl - min_wl) / dl))
    n_spatial_blocks = int(math.ceil(img.shape[0]/vert_size))
    logger.info("Using %d spatial and %d wavelength blocks" % (n_spatial_blocks, n_wl_blocks))

    data = []
    scaling = numpy.zeros((n_wl_blocks, n_spatial_blocks,3))
    for i_wl, i_spatial in itertools.product(range(n_wl_blocks), range(n_spatial_blocks)):
        
        #
        # Cut out strips in spatial direction
        #
        y_min = i_spatial * vert_size
        y_max = numpy.min([y_min+vert_size, img.shape[0]])
        
        strip_img = img[y_min:y_max]
        strip_sky = sky[y_min:y_max]
        strip_wl = wl[y_min:y_max]

        #
        # Now select wavelength interval
        #
        wl_min = i_wl*dl + min_wl
        wl_max = wl_min + dl
        in_wl_range = (strip_wl >= wl_min) & (strip_wl <= wl_max)

        sel_img = strip_img[in_wl_range]
        sel_sky = strip_sky[in_wl_range]
        
        simple_median = bottleneck.nanmedian(sel_img/sel_sky)
        simple_mean = bottleneck.nanmean(sel_img/sel_sky)
        weight_mean = bottleneck.nansum(sel_img) / bottleneck.nansum(sel_sky)
        # weighted mean = sum(img/sky * sky)/sum(sky) where sky=weight
        data.append([i_wl, i_spatial, simple_mean, simple_median, weight_mean])

        scaling[i_wl, i_spatial,:] = [simple_mean, simple_median, weight_mean]

    data = numpy.array(data)

    #
    # Now do some 2-d filtering and interpolating
    #
    filtered = numpy.zeros(scaling.shape)
    for plane in range(3):
        padded = numpy.zeros((scaling.shape[0]+2*smooth, scaling.shape[1]+2*smooth))
        padded[:,:] = numpy.NaN
        padded[smooth:-smooth, smooth:-smooth] = scaling[:,:,plane]
        
        
        for y,x in itertools.product(range(scaling.shape[0]), range(scaling.shape[1])):
            filtered[y,x,plane] = bottleneck.nanmedian(padded[y:y+2*smooth+1, x:x+2*smooth+1])

        pyfits.HDUList([
            pyfits.PrimaryHDU(),
            pyfits.ImageHDU(data=scaling[:,:,plane].T,name="IN"),
            pyfits.ImageHDU(data=filtered[:,:,plane].T,name="out"),
            ]).writeto("scaling_%d.fits" % (plane), clobber=True)
        
    
    interpol = scipy.interpolate.RectBivariateSpline(
        x=numpy.arange(n_wl_blocks)*dl+min_wl,
        y=numpy.arange(n_spatial_blocks)*vert_size,
        z=filtered[:,:,1],
        s=0
        #kind='linear',
        #bounds_error=False,
        #fill_value=numpy.NaN,
        )
    
    y,_ = numpy.indices(img.shape)
    full2d = interpol(wl.ravel(), y.ravel(), grid=False).reshape(img.shape)
    pyfits.PrimaryHDU(data=full2d).writeto("scale2d.fits", clobber=True)

    return data, filtered, full2d


def minimize_sky_residuals(img, sky, vert_size=5, smooth=20, debug_out=True):

    print img.shape, sky.shape

    n_slices = int(math.ceil(img.shape[0] / float(vert_size)))
    print n_slices
    
    scaling_data = numpy.zeros((n_slices,5))

    for curslice in range(n_slices):
        
        y0 = curslice * vert_size
        y1 = y0+vert_size if ( y0+vert_size < img.shape[0]) else img.shape[0]
        print y0,y1

        img_slice = img[y0:y1, :]
        sky_slice = sky[y0:y1, :]

        p_init = [1.0, 0.0, 0.0]

        fit_args = (img_slice, sky_slice)
        _fit = scipy.optimize.leastsq(
            sky_residuals,
            p_init, 
            args=fit_args,
            maxfev=500,
            full_output=1)
        #print _fit[0]

        # img_slice -= scaled_sky(_fit[0],sky_slice)

        scaling_data[curslice, 0] = 0.5*(y0+y1)
        scaling_data[curslice, 1:4] = _fit[0]


    #
    # Now fit a low-order spline to the scaling profile
    #
    medfilt = scipy.ndimage.filters.median_filter(
        scaling_data[:,1],
        smooth,
        mode='wrap',
        )
    scaling_data[:,-1] = medfilt[:]
    print medfilt

    interp = scipy.interpolate.InterpolatedUnivariateSpline(
        x=scaling_data[:,0],
        y=scaling_data[:,-1],
        k=3,
#        bounds_error=False,
#        fill_value=0,
        )

    full_profile = interp(numpy.arange(img.shape[0]))

    if (debug_out):
        numpy.savetxt("optscale.full", numpy.append(
            numpy.arange(full_profile.shape[0]).reshape((-1,1)),
            full_profile.reshape((-1,1)), axis=1))
        
        numpy.savetxt("optscale.out", scaling_data)

    return scaling_data, full_profile.reshape((-1,1))



if __name__ == "__main__":

    img_fn = sys.argv[1]
    img_hdu = pyfits.open(img_fn)

    sky_fn = sys.argv[2]
    sky_hdu = pyfits.open(sky_fn)

    img = img_hdu[0].data

    sky = sky_hdu[0].data

    full_profile = minimize_sky_residuals(img, sky, vert_size=5, smooth=20, debug_out=True)

    skysub = img - (sky * full_profile)
    pyfits.PrimaryHDU(data=img).writeto(sys.argv[3], clobber=True)
