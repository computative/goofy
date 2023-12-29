t0 = time()
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
lambda = parse(Float64, ARGS[3])
order = parse(Int64, ARGS[4]); degree = parse(Int64, ARGS[5])
lens = parse.(Int, split(chop(ARGS[6]; head=1, tail=1), ','))
vol = parse(Int64, ARGS[7])
intercept = false


# it is assumed that the order of the orbitals in the sub-blocks are the following:
# first come all the s-orbitals, then all the p-orbitals, then the d-orbitals and so on



# H, R are collections of configurations, Z and cell are charge and unit cell for all configs

for len in lens
    _H = []; _R = []; _unitcell = []; _Z = []; _IJ = []
    
    for (i, file_id) in enumerate(file_ids)
        path = "/home/marius/Dokumenter/Skole/phd/goofy.git/test/data1.h5"
        chosen = random_idx(path, vol*len, rcut)
        IJ = chosen[:,1]
        idx = chosen[:,2]
        H, R, cell, Z = parse_files(path, IJ, idx)
        append!(_R,[R]); append!(_H,[H]); append!(_unitcell,[cell]); append!(_Z,[Z]); append!(_IJ,[IJ]); 
    end


    L_cfg = Dict(0=>1, 1=>1)  # r0cut rcut
    ace_param = [degree, order, rcut, renv, L_cfg]
    fit_param = [_H[1], lambda, lsqr_solver, intercept]
    system = [_IJ[1], _R[1], _Z[1], _unitcell[1]]
    c, fitted, residuals, basis, configs = train(system, ace_param, fit_param)


    #retucer takes in vector of blocks where the statistic has been applied
    
    function rmse(X::Vector{Matrix{ComplexF64}}, Y::Vector{Matrix{ComplexF64}}) 
        Z = map( (x,y) -> x .- y, X, Y )
        N = length(Z)*length(first(Z))
        return ( real(dot( Z, Z ))/N )^0.5
    end
    
    function rel(X::Vector{Matrix{ComplexF64}}, Y::Vector{Matrix{ComplexF64}}) 
        Z = map( (x,y) -> x./ (y .+ 0) .- 1, X, Y )
        N = length(Z)*length(first(Z))
        return ( real(dot( Z, Z ))/N )^0.5
    end

    function res(X::Vector{Matrix{ComplexF64}}, Y::Vector{Matrix{ComplexF64}})
        N = length(X)*length(first(X))
        residuals = map( (x,y) -> norm(x .- y)/N^0.5, X, Y )
        return residuals
    end
    

    env = CylindricalBondEnvelope(rcut, renv, rcut/2)
    test_configs = coords2configs([_IJ[2], _R[2]], _Z[2], env, _unitcell[2])

    rmse_jig = test_setup(c, basis, rmse)
    relerr_jig = test_setup(c, basis, rel_err)

    #result = Dict( "order" => order, "degree" => degree, "lambda" => lambda, 
    #    "nobs" => vol*len, "rmse" => test_jig(rmse, H), 
    #    "rel_err" => test_jig(rel_err, H), "res" => test_jig(res, H)
    #)
    
    println("train")
    println( JSON.json(Dict( "rmse" => vec(rmse_jig( _H[1],configs)), "rel_err" => vec(relerr_jig(_H[1],configs)) )))
    
    println("test")
    println( JSON.json(Dict( "rmse" => vec(rmse_jig(_H[2],test_configs)), "rel_err" => vec(relerr_jig(_H[2],test_configs)) )))
    
    #rmse_train = test(c, basis, configs, _H[1], absolute, reducer)
    #rmse_test = test(c, basis, test_configs, _H[2], absolute, reducer)
    #println("rmse ", rmse_train, " ", "train ", lambda, " ",  order , " ",  degree , " " , len, " " , vol)
    #println("rmse ", rmse_test, " ", "test ", lambda, " ",  order , " ",  degree , " " , len, " " , vol)
    #
    #gabor_train = test(c, basis, configs, _H[1], relative, reducer)
    #gabor_test = test(c, basis, test_configs, _H[2], relative, reducer)
    #println("gabo ", gabor_train, " ", "train ", lambda, " ",  order , " ",  degree , " " , len, " " , vol)
    #println("gabo ", gabor_test, " ", "test ", lambda, " ",  order , " ",  degree , " " , len, " " , vol)
end
t1 = time()
println("Elapsed  $( Int64(round(t1-t0)) )s")