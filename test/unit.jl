using goofy, LinearAlgebra


rcut = 5.0
n = 100
path = "/home/marius/Dokumenter/Skole/phd/goofy-e2e-data"

chosen = random_idx(path * "/e2e.h5", n, rcut)
IJ = chosen[:,1]
idx = chosen[:,2]
H, R, cell, Z = parse_files(path * "/e2e.h5", IJ, idx )

pass1 = reduce(&, [ norm(r[J,:] - r[I,:]) < rcut for (r, (I, J) ) in zip(R', IJ) ])

if !pass1
    @show [ norm(r[J,:] - r[I,:]) < rcut for (r, (I, J) ) in zip(R', IJ) ]
    @show [ norm(r[J,:] - r[I,:])  for (r, (I, J) ) in zip(R', IJ) ]
    @show [ (r[J,:] - r[I,:]) < rcut for (r, (I, J) ) in zip(R', IJ) ]
end

_H, _R, _IJ, _cell, _Z = parse_files_depricated(path, "1", n, 1, IJ, rcut)
atol = 1e-14


pass2 = (norm(_H - H) + norm(_R - R) + norm(_cell - cell) + norm(_Z - Z) < 4*atol)

if !pass2
    @show norm(_H - H)
    @show norm(_R - R)
    @show norm(_cell - cell)
    @show norm(_Z - Z)
end


pass = pass1 & pass2
@show pass
