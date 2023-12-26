using goofy, JSON, HDF5, LinearAlgebra, Random

# I set the degree, maxorder, cutoff and regularization parameter
rcut = 5.0; renv = rcut/2; 
order = 1; degree = 5
lambda = 1e-12
n = 4
path = "/home/marius/Dokumenter/Skole/phd/goofy.git/test/data7.h5"

# H, R are collections of configurations, Z and cell are charge and unit cell for all configs



chosen = random_idx(path, n, rcut)
IJ = chosen[:,1]
idx = chosen[:,2]
H, R, cell, Z = parse_files(path, IJ, idx )


L_cfg = Dict(0=>1, 1=>1)  # r0cut rcut
ace_param = [degree, order, rcut, renv, L_cfg]
fit_param = [H, lambda, inv_solver, false]
system = [IJ, R, Z, cell]




c, fitted, residuals, basis, configs = train(system, ace_param, fit_param)

absolute(X::Matrix{ComplexF64},Y::Matrix{ComplexF64}) = X - Y
relative(X::Matrix{ComplexF64},Y::Matrix{ComplexF64}) = X./(Y .+ 1e-7).-1

println( "rmse", test(c, basis, configs, H, absolute) )
println( "gabo", test(c, basis, configs, H, relative) )

