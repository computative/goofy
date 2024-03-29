using goofy, JSON, HDF5, LinearAlgebra, Random
using ACE: BondEnvelope, CylindricalBondEnvelope
using Dates

# learning curves from the terminal
# execute with julia --project=. test/learning.jl r l o d l   
# rcut lambda order degree [len]


# basis config
# I set the degree, maxorder, cutoff and regularization parameter

rcut = parse(Float64, ARGS[1])
renv = rcut/2; 
lambda = parse(Float64, ARGS[2])
order = parse(Int64, ARGS[3]); degree = parse(Int64, ARGS[4])
lens = parse.(Int, split(chop(ARGS[5]; head=1, tail=1), ','))

# it is assumed that the order of the orbitals in the sub-blocks are the following:
# first come all the s-orbitals, then all the p-orbitals, then the d-orbitals and so on


# H, R are collections of configurations, Z and cell are charge and unit cell for all configs

for len in lens
    _H = []; _R = []; _unitcell = []; _Z = []; _IJ = []
    chosen = []
    path = ""
    for _ in 1:2
        path = abspath(@__DIR__, 
                "../../goofy.files/data/structures/structure_diam_bulk333_300K/_dft_1.h5")
        IJ, idx = random_idx(path, len, rcut)
        append!(chosen,[(IJ, idx)])
        H, R, cell, Z = parse_files(path, IJ, idx)
        append!(_R,[R]); append!(_H,[H]); append!(_unitcell,[cell]); 
        append!(_Z,[Z]); append!(_IJ,[IJ]); 
    end
    L_cfg = Dict(0=>1, 1=>1)  # r0cut rcut
    ace_param = [degree, order, rcut, renv, L_cfg]
    fit_param = [_H[1], lambda, lsqr_solver]
    system = [_IJ[1], _R[1], _Z[1], _unitcell[1]]
    c, fitted, residuals, basis, configs = train(system, ace_param, fit_param)

    label = join(split(split(path, "/")[end-1], "_")[2:end-1], "-")
    timestamp   = Dates.format(now(), "dd-mm-yyTHH-MM-SS")
    outpath = abspath(@__DIR__, "../../goofy.files/models/L")
    write_item((basis,c),outpath * string(len) * "-" *  label * "-" * timestamp * ".mdl")
    write_item(chosen[1],outpath * string(len) * "-" *  label * "-" * timestamp * ".cho")

end