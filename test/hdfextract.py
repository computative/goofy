import h5py
import numpy as np
import sys

inpath = sys.argv[1]
extracted =  eval(sys.argv[2])
outpath = sys.argv[3]

with h5py.File(inpath, "r") as hdfr:
    with h5py.File(outpath, "w" ) as hdfw:
        for item in extracted:
            matrix = hdfr[str(item)][:,:]
            hdfw.create_dataset( str(item), data=matrix)

