    using goofy, JSON, HDF5, LinearAlgebra, Random

    # I set the degree, maxorder, cutoff and regularization parameter
    rcut = 5.0; renv = rcut/2; 
    order = 1; degree = 20
    lambda = 1e-20
    len = 2


    # H, R are collections of configurations, Z and cell are charge and unit cell for all configs

    H, R, IJ, cell, Z = parse_files("/home/marius/Dokumenter/Skole/phd/goofy.git/test", "1", len)
    L_cfg = Dict(0=>1, 1=>1)  # r0cut rcut
    ace_param = [degree, order, rcut, renv, L_cfg]
    fit_param = [H, lambda, "none"]
    system = [IJ, R, Z, cell]


    c, fitted, residuals, basis, configs = train(system, ace_param, fit_param)
    stat1 = test(c, basis, configs, H, "rmse")
    stat2 = test(c, basis, configs, H, "gabor")

    println( "rmse", stat1 )
    println( "gabo", stat2 )

