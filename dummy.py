#!/usr/bin/env python

import os, sys
import wlcal, numpy
import pyfits

from wlcal import rssmodelwave


if __name__ == "__main__":

    fitsfile = sys.argv[1]

    hdulist = pyfits.open(fitsfile)
    primhdr = hdulist[0].header
    
    rbin,cbin = numpy.array(primhdr["CCDSUM"].split(" ")).astype(int)

    grating = primhdr['GRATING'].strip()
    grang = float(primhdr['GR-ANGLE'])
    artic = float(primhdr['CAMANG'])
    
    #arc_rc = hdulist['SCI'].data
    #rows,cols = arc_rc.shape
    rows,cols = [256,512]

    cols = 2048

    lam_c = rssmodelwave(grating,grang,artic,cbin,cols)
    print lam_c.shape

    print lam_c

    x = numpy.arange(cols)
    
    merged = numpy.append(x.reshape((-1,1)),
                          lam_c.reshape((-1,1)), axis=1)
    numpy.savetxt("lambda_vs_position.dat", merged)
