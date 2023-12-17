from sklearn.neighbors import KernelDensity
import numpy as np
import sys, os
from scipy.stats import wasserstein_distance as emd

n = 200

bw = 1
samples = 100

__path__ = os.path.dirname(__file__)
files = [ file for file in os.listdir(__path__) if ".dat" in file ]
materials = [ file.replace(".dat", "") for file in files]

title = { "bulk":"bulk", "vacancy":"vacancy", "rotsurf":"rotated surface", "surface":"surface" }

for file,material in zip(files,materials):
    

    f = open(__path__ + "/" + file)
    data = f.read().strip()
    f.close()
    exact = []
    for line in data.split("\n"):
        words = line.strip().split(" ")
        mode = words[0]
        lst = eval( " ".join(words[1:]) )
        if mode == "Eexact":
            exact = np.array(lst).reshape((len(lst),1))
            break
        else:
            exit("There's a line that neither is a prediction nor exact.")
    a = sorted(exact)[0]
    b = sorted(exact)[-1]
    k = len(exact)

    f = open(__path__ + "/" + file)
    data = f.read().strip()
    f.close()

    predictions = []
    for line in data.split("\n"):
        words = line.strip().split(" ")
        mode = words[0]
        lst = eval( " ".join(words[1:]) )
        if mode == "Epredict":
           predictions.append( np.array(lst).reshape((k,1)) )

    # vi har en liste med lister bestående av alle Epredict

    e = np.linspace(a,b,n)
    e = e.reshape(len(e),1)

    curves = []
    for prediction in predictions:
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(prediction)
        density = np.exp(kde.score_samples(e))
        curves.append(density)


    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(exact)
    exact_density = np.exp(kde.score_samples(e))

    e = np.reshape(e,(n,1)).T[0]

    abs_mu = []
    abs_std_h = []
    abs_std_l = []
    relative = []

    curves = np.array(curves)

    for i, e_i in enumerate(e):
        c_i = sorted(curves[:,i])
        abs_mu.append( sorted(np.abs((c_i - exact_density[i])) )[ int(np.round(0.5*samples)) ] )
        abs_std_h.append( sorted(np.abs((c_i - exact_density[i])) )[ int(np.round(0.95*samples)) ] )
        abs_std_l.append( sorted(np.abs((exact_density[i] - c_i)) )[ int(np.round(0.05*samples)) ] )



    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.plot(e, np.log10(abs_mu ), c="k")
    ax.plot(e, np.log10(abs_std_h ), c="k",linewidth=0.6)
    ax.plot(e, np.log10(abs_std_l ), c="k",linewidth=0.5)
    ax.set_title(f"DOS-error 95%-confidence interval ({ title[material].capitalize() })")
    ax.set_ylabel("log10(absolute error)")
    ax.set_xlabel("Spectrum [eV]")
    #fig.show()
    fig.savefig(f"{ __path__ }/bw_{ bw }_{material}.png")
    #input()

