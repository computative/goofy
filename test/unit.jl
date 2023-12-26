using goofy, LinearAlgebra


rcut = 5.0
n = 20
path = "/home/marius/Dokumenter/Skole/phd/goofy-e2e-data"

chosen = random_idx(path * "/e2e.h5", n, rcut)
IJ = chosen[:,1]
idx = chosen[:,2]
H, R, cell, Z = parse_files(path * "/e2e.h5", IJ, idx )

_H, _R, _IJ, _cell, _Z = parse_files_depricated("/home/marius/Dokumenter/Skole/phd/goofy-e2e-data", "1", n, 1, IJ, rcut)


atol = 1e-14
pass = (norm(_H - H) + norm(_R - R) + norm(_cell - cell) + norm(_Z - Z) < 4*atol)

@show pass
