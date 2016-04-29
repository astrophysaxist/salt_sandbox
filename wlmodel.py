#!/usr/bin/env python

import os, sys
import pyfits
import numpy
import scipy

import pysalt

datadir="/work/salt/sandbox_official/polSALT/polsalt/data/"

# based on http://www.sal.wisc.edu/PFIS/docs/rss-vis/archive/protected/pfis/3170/3170AM0010_Spectrograph_Model_Draft_2.pdf
# and https://github.com/saltastro/SALTsandbox/blob/master/polSALT/polsalt/specpolmap.py

def rssmodelwave(#grating,grang,artic,cbin,refimg,
        header, img,
        xbin=1, ybin=1):
#   compute wavelengths from model (this can probably be done using pyraf spectrograph model)

    ncols = img.shape[0]
    nrows = img.shape[1]

    # compute X/Y position for each pixel
    y,x = numpy.indices(img.shape)
    # also account for binning
    y *= ybin
    x *= xbin

    #
    #
    # Load spectrograph parameters
    #
    spec=numpy.loadtxt(datadir+"spec.txt",usecols=(1,))
    grating_rotation_home_error = spec[0]

    Grat0,Home0,ArtErr,T2Con,T3Con=spec[0:5]
    FCampoly=spec[5:11]

    grating_names=numpy.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    #grname=numpy.loadtxt(datadir+"gratings.txt",dtype=str,usecols=(0,))
    grlmm,grgam0=numpy.loadtxt(datadir+"gratings.txt",usecols=(1,2),unpack=True)


    #
    # Load all necessary information from FITS header
    #
    grating_angle = header['GR-ANGLE'] # alpha_C
    articulation_angle = header['CAMANG'] #GRTILT'] # A_C
    grating_name = header['GRATING']

    print "grating-angle:", grating_angle
    print "articulation angle:", articulation_angle
    print "grating name:", grating_name

    
    # get grating data: lines per mm
    #grnum = numpy.where(grname==grating)[0][0]
    #lmm = grlmm[grnum]
    grnum = numpy.where(grating_names==grating_name)[0][0]
    grating_lines_per_mm = grlmm[grating_name == grating_names][0]
    print "grating lines/mm:", grating_lines_per_mm

    #alpha_r = numpy.radians(grang+Grat0)
    alpha_r = numpy.radians(grating_angle+Grat0)
    #beta0_r = numpy.radians(artic*(1+ArtErr)+Home0)-alpha_r
    beta0_r = numpy.radians(articulation_angle*(1+ArtErr)+Home0)-alpha_r
    gam0_r = numpy.radians(grgam0[grnum])

    print "alpha-r:", alpha_r
    print "beta_r :", beta0_r
    print "gamma_r:", gam0_r

    # compute reference wavelength at center of focal plane
    #lam0 = 1e7*numpy.cos(gam0_r)*(numpy.sin(alpha_r) + numpy.sin(beta0_r))/lmm
    lam0 = 1e7*numpy.cos(gam0_r)*(numpy.sin(alpha_r) + numpy.sin(beta0_r))/grating_lines_per_mm
    print "reference wavelength:", lam0

    # compute camera focal length
    ww = (lam0-4000.)/1000.
    fcam = numpy.polyval(FCampoly,ww)
    print "camera focal length @ 4000A:", fcam,"mm"

    # compute dispersion per pixel
    disp = (1e7*numpy.cos(gam0_r)*numpy.cos(beta0_r)/grating_lines_per_mm) / (fcam/.015)
    #disp = (1e7*numpy.cos(gam0_r)*numpy.cos(beta0_r)/lmm)/(fcam/.015)
    print "dispersion:", disp," angstroems/pixel"


    # 
    # Iteratively compute a lambda for each pixel, refine the focal length as 
    # fct of lambda, and recompute lambda
    # 
    _x = (x - 3162) * 0.015 #/ 3162.
    _y = (y - 2048) * 0.015 # / 2048.
    print numpy.min(x), numpy.max(x)
    print numpy.min(y), numpy.max(y)

    print numpy.min(_x), numpy.max(_x)
    print numpy.min(_y), numpy.max(_y)

    alpha = numpy.ones(img.shape) * alpha_r
    for iteration in range(4):
        beta = _x/fcam + beta0_r 
        gamma = _y/fcam + gam0_r
        print beta.shape, gamma.shape

        # compute lambda (1e7 = angstroem/mm)
        _lambda = 1e7 * numpy.cos(gamma) * (numpy.sin(beta) + numpy.sin(alpha)) / grating_lines_per_mm

        L = (_lambda - 4000.) / 1000.
        fcam = numpy.polyval(FCampoly,L)
        print "ITER", iteration, fcam.shape

        pyfits.PrimaryHDU(data=_lambda).writeto("lambda_%d.fits" % (iteration+1), clobber=True)

        
    return _lambda

        
    # now compute F_cam as function of lambda_0
    # use polynomial fit from ZEMAX camera model (that's step from Ken's docu)
    dfcam = 3.162*disp*numpy.polyval([FCampoly[x]*(5-x) for x in range(5)],ww)


    #T2 = -0.25*(1e7*numpy.cos(gam0_r)*numpy.sin(beta0_r)/lmm)/(fcam/47.43)**2 + T2Con*disp*dfcam
    T2 = -0.25*(1e7*numpy.cos(gam0_r)*numpy.sin(beta0_r)/grating_lines_per_mm)/(fcam/47.43)**2 + T2Con*disp*dfcam
    T3 = (-1./24.)*3162.*disp/(fcam/47.43)**2 + T3Con*disp
    T0 = lam0 + T2 
    T1 = 3162.*disp + 3*T3

    return


    # compute normalized X-position (range [-1,1], X=0 is center of middle chip)
    X = (numpy.array(range(cols))+1-cols/2)*cbin/3162.

    lam_X = T0+T1*X+T2*(2*X**2-1)+T3*(4*X**3-3*X)
    return lam_X


if __name__ == "__main__":
    
    fn = sys.argv[1]
    hdulist = pyfits.open(fn)

    binning = pysalt.get_binning(hdulist)
    xbin, ybin = 4,4
    print "binning", binning, xbin, ybin

    rssmodelwave(#grating,grang,artic,cbin,refimg,
        header=hdulist[0].header, 
        img=hdulist['SCI'].data,
        xbin=xbin, ybin=ybin)

    
