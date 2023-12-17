import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# __file__, os.path.dirname, os.listdir

__path__ = os.path.dirname(__file__)
files = [ file for file in os.listdir(__path__) if ".dat" in file ]
materials = [item.replace(".dat", "").replace("e_","") for item in files]
gaps = [-7.36,-7.49,-8.15,-7.0]

for bandgap,material,file in zip(gaps, materials, files):
    Eexact = []
    Epredict = []
    yticks = []
    xticks = []
    with open(file) as f:
        i = 1
        for line in f:
            items = line.strip().split(" ")
            mode = items[0]
            length = eval(items[1])
            lst = eval( " ".join(items[2:]) )
            eval(mode).append(lst)
            i += 1
            if length != i//2:
                error("E: I think the input file has lines in the wrong order. Expected length = {length}, got {i//2}")
    Epredict = np.array(Epredict)
    Eexact = np.array(Eexact)

    

    fig, ax  = plt.subplots(figsize=(4,6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.imshow(np.log10(np.abs(Epredict - Eexact)), aspect="auto")
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    n = 3

    gapidx = np.argmax(Eexact - bandgap > 0)
    E = np.array(Eexact[0])

    if len(E)//2 < gapidx: # gap er i øvre halvplan
        span = np.floor(gapidx/n)
    else:
        span = np.floor((len(E)-gapidx)/n)
    
    npts_h = int(np.floor((len(E)-gapidx)/span))
    npts_l = int(np.floor(gapidx/span))
    
    idxs = [int(gapidx + i*span) for i in range( npts_h +1 )] + \
       [int(gapidx - i*span) for i in range(1, npts_l+1 )] 
    E_ticklabels = np.round(E[idxs], 1).astype(str)
    #E_ticklabels[0] = f"{ round(E[idxs[0]],1) }"
    ax.set_xticks( idxs )
    ax.set_xticklabels( E_ticklabels )
    #ax.axvline(x=idxs[0],color='red')
    
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Training set size")
    ax.set_title(f"{material.capitalize()}: log10 of absolute error")
    fig.savefig(f"{__path__}/img_{material}.png")




