import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import wasserstein_distance as emd
# __file__, os.path.dirname, os.listdir

__path__ = os.path.dirname(__file__)
files = [ file for file in os.listdir(__path__) if ".dat" in file ]
materials = [item.replace(".dat", "").replace("e_","") for item in files]

title = { "bulk":"bulk", "surf":"surface", "rotsurf":"rotated surface", "vacancy":"vacancy" }

for material,file in zip(materials, files):
    Eexact = []
    Epredict = []
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
    Eexact = np.array(Eexact[0])

    

    fig, ax  = plt.subplots()

    x = np.arange(1,101)
    y = []

    for i in range(len(x)):
        y.append(emd(Epredict[i],Eexact))

    ax.plot( x, y )



    
    ax.set_xlabel("Data size")
    ax.set_ylabel("EM distance")
    ax.set_title(f"Wasserstein metric of DOS (exact vs predicted) for {title[material]}")
    ax.set_yscale('log')
    fig.savefig(f"{__path__}/img_dos_{material}.png")
    #fig.show()
    #input()
    #exit()



