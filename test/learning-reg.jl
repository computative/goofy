using goofy, JSON, HDF5, LinearAlgebra, Random
using ACE: BondEnvelope, CylindricalBondEnvelope

# learning curves from the terminal
# execute with julia --project=. test/learning.jl n m l u v w k  (where n and m are the id of the training and testing data. 
# l is regularization parameter, u and v are the order and degree respectively
# While k is the number of observations that should be drawn from each full size hamiltonian matrix and w is the number of matrces to sample


# basis config
# I set the degree, maxorder, cutoff and regularization parameter
rcut = 5.0; renv = rcut/2; 
file_ids = ARGS[1:2]
lambda = 2.0 .^ Vector(-60:-10)
order = parse(Int64, ARGS[4]); degree = parse(Int64, ARGS[5])
len = 95
vol = parse(Int64, ARGS[7])

# it is assumed that the order of the orbitals in the sub-blocks are the following:
# first come all the s-orbitals, then all the p-orbitals, then the d-orbitals and so on




for _lambda in lambda
    println(_lambda)
    _H = []; _R = []; _unitcell = []; _Z = []; _IJ = []
    for (i, file_id) in enumerate(file_ids)
        H, R, IJ, cell, Z = parse_files("/home/marius/Dokumenter/Skole/phd/goofy.git/test", file_id, len)
        append!(_R,[R]); append!(_H,[H]); append!(_unitcell,[cell]); append!(_Z,[Z]); append!(_IJ,[IJ]); 
    end 

    #_IJ[1] = [(2, 3), (3, 1), (2, 3)]
    L_cfg = Dict(0=>1, 1=>1)  # r0cut rcut
    ace_param = [degree, order, rcut, renv, L_cfg]
    fit_param = [_H[1], _lambda, "none"]
    system = [_IJ[1], _R[1], _Z[1], _unitcell[1]]
    c, fitted, residuals, basis, configs = train(system, ace_param, fit_param)

    rmse_train = test(c, basis, configs, _H[1], "rmse")
    rmse_test = test(c, basis, coords2configs([_IJ[2], _R[2]], _Z[2], CylindricalBondEnvelope(rcut, renv, rcut/2), _unitcell[2]), _H[2], "rmse")

    println("rmse ", rmse_train, " ", "train ", "lambda", " ",  order , " ",  degree , " " , len, " " , vol)
    println("rmse ", rmse_test, " ", "test ", "lambda", " ",  order , " ",  degree , " " , len, " " , vol)

    gabor_train = test(c, basis, configs, _H[1], "gabor")
    gabor_test = test(c, basis, coords2configs([_IJ[2], _R[2]], _Z[2], CylindricalBondEnvelope(rcut, renv, rcut/2), _unitcell[2]), _H[2], "gabor")

    println("gabo ", gabor_train, " ", "train ", "lambda", " ",  order , " ",  degree , " " , len, " " , vol)
    println("gabo ", gabor_test, " ", "test ", "lambda", " ",  order , " ",  degree , " " , len, " " , vol)
end


