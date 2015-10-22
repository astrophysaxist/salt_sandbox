#!/usr/bin/env python

import pickle, numpy, pyfits, os, sys
import traceline

arclist = pickle.load(open(sys.argv[1], "rb"))

print arclist.shape

trace_every = float(sys.argv[2])
min_line_separation = float(sys.argv[3])
n_pixels = 3160

lines = traceline.pick_line_every_separation(
    arclist,
    trace_every, min_line_separation,
    n_pixels)

print lines
