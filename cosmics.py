#!/usr/bin/env python

import os, sys, podi_cython, pyfits, numpy

if __name__ == "__main__":

    hdulist = pyfits.open(sys.argv[1])

    data = hdulist['SKYSUB.OPT'].data

    gain = 1.5
    readnoise = 6

    sigclip = 5.0
    sigfrac = 0.3
    objlim = 5.0
    saturation_limit=65000

    crj = podi_cython.lacosmics(data.astype(numpy.float64), 
                          gain=gain, 
                          readnoise=readnoise, 
                          niter=3,
                          sigclip=sigclip, sigfrac=sigfrac, objlim=objlim,
                          saturation_limit=saturation_limit,
                          verbose=False)

    cell_cleaned, cell_mask, cell_saturated = crj

    hdulist.append(pyfits.ImageHDU(data=cell_cleaned,
                                   header=hdulist['SCI.RAW'].header,
                                   name="PODI.CRJ"))

    hdulist.writeto(sys.argv[1][:-5]+".odicrj.fits", clobber=True)
