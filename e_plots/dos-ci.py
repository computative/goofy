import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import wasserstein_distance as emd
# __file__, os.path.dirname, os.listdir


__path__ = os.path.dirname(__file__)
materials = ["bulk","surface","rotsurf","vacancy"]

title = { "bulk":"bulk", "surf":"surface", "rotsurf":"rotated surface", "vacancy":"vacancy" }
n = 100 # how many training sets?
k = 20 # how many files will make up the sample used for the CI?
x = np.arange(1,n+1)
#x = np.zeros((k,n))
for material in materials:
    ys = []
    files = [ filename for filename in os.listdir(f"{__path__}/new_data_format") if material in filename ]
    for j, filename in enumerate(files):
        Eexact = []
        Epredict = []
        with open(f"{__path__}/new_data_format/{filename}") as f:
            i = 1
            for line in f:
                items = line.strip().split(" ")
                mode = items[0]
                length = eval(items[1])
                lst = eval( " ".join(items[4:]) )
                eval(mode).append(lst)
                i += 1
                if length != i//2:
                    error("E: I think the input file has lines in the wrong order. Expected length = {length}, got {i//2}")
        Epredict = np.array(Epredict)
        Eexact = np.array(Eexact[0])
        
        #x[j] = np.arange(1, n+1) + (np.random.rand(n)-0.5)
        
        y = []
        for i in range(n):
            y.append(emd(Epredict[i],Eexact))
        ys.append(y)

ys = np.array(ys)

upr = []
mu = []
lwr = []

for i, xi in enumerate(x):
    slce = sorted(ys[:,i])
    #print( int(np.floor(0.95*k)) )
    upr.append( slce[ int(np.floor(0.95*k)) ] )
    lwr.append( slce[ int(np.ceil(0.05*k)) ] )
    mu.append( slce[ int(np.round(0.5*k)) ] )

fig, ax  = plt.subplots()
ax.plot( x, upr , "k:")
ax.plot( x, lwr , "k:")
ax.plot( x, mu , "k")

ax.set_xlabel("Data size")
ax.set_ylabel("EM distance")
ax.set_title(f"Wasserstein metric of DOS (exact vs predicted) for {title[material]}")
ax.set_yscale('log')
fig.savefig(f"{__path__}/img_dos_ci_{material}.png")


"""
y = np.log10(ys.reshape((k*n)))
X = x.reshape((k*n,1))

from pygam import LinearGAM, s, InvGaussGAM, PoissonGAM

# GammaGAM
gam = LinearGAM(s(0, n_splines=4,spline_order=2)).gridsearch(X, y) # virker som n_splines bestemmer hvor mange ganger kurva får lov til å svinge over x-aksen
XX = gam.generate_X_grid(term=0, n=500) #virker som n bestemmer hvor mange punkter vi plotter langs x-aksen

plt.plot(XX, gam.predict(XX), 'k')
#plt.plot(XX, gam.confidence_intervals(XX, width=.95), 'k:')
plt.plot(XX, gam.prediction_intervals(XX, width=.95), 'k:', linewidth=0.5)

#plt.scatter(X, y, facecolor='gray', edgecolors='none')
plt.title('95% prediction interval');
plt.show()
"""
