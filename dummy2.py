#!/usr/bin/env python

import numpy, os, sys
import wlcal
import pyfits

if __name__ == "__main__":

    fits = sys.argv[1]
    
    hdu = pyfits.open(fits)

    ncols = 3100
    
    model = wlcal.KenRSSModel(hdu[0].header, ncols)

    print 
    grang = hdu[0].header['GR-ANGLE']
    print grang

    angs = numpy.linspace(grang-5, grang+5, 101)
    for idx, ang in enumerate(angs):
        model.compute(grang=ang)
        numpy.savetxt(sys.argv[2]+"--%.2f" % (ang), model.get_wavelength_list())

    artic=float(hdu[0].header['CAMANG'])
    artics = numpy.linspace(artic-5, artic+5, 101)
    for idx, artic in enumerate(artics):
        model.compute(artic=artic)
        numpy.savetxt(sys.argv[2]+"-artic--%.2f" % (artic), model.get_wavelength_list())
