import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
from pygam.datasets import mcycle

_, y = mcycle(return_X_y=True)

print(y)

X = np.array(np.arange(len(y) ) ).reshape((len(y),1))




#gam = LinearGAM(n_splines=50).gridsearch(X, y)
gam = LinearGAM(s(0, n_splines=100,spline_order=2)).gridsearch(X, y) # virker som n_splines bestemmer hvor mange ganger kurva får lov til å svinge over x-aksen
XX = gam.generate_X_grid(term=0, n=500) #virker som n bestemmer hvor mange punkter vi plotter langs x-aksen

plt.plot(XX, gam.predict(XX), 'k')
plt.plot(XX, gam.prediction_intervals(XX, width=.95), 'k:')

plt.scatter(X, y, facecolor='gray', edgecolors='none')
plt.title('95% prediction interval');
plt.show()
