using goofy, JSON, HDF5, LinearAlgebra, Random
using ACE: BondEnvelope, CylindricalBondEnvelope
using Dates

# execute with julia --project=. test/learning.jl r l p p p p
# rcut [len] path_test path_train path_choice path_model


# basis config
# I set the degree, maxorder, cutoff and regularization parameter

rcut = parse(Float64, ARGS[1])
renv = rcut/2; 
len = parse(Int64, ARGS[2])

path_test = ARGS[3]
path_train = ARGS[4]
path_choice = ARGS[5]
path_model = ARGS[6]

basis,c = read_item(path_model)


_H = []; _configs = []; _IJ = []; _idx = []

IJ, idx = read_item(path_choice)
append!(_IJ,[IJ]); append!(_idx,[idx]); 
IJ, idx = random_idx(path_test, len, rcut)
append!(_IJ,[IJ]); append!(_idx,[idx]); 

for (IJ,idx,path) in zip(_IJ,_idx,[path_train,path_test])
    H, R, cell, Z = parse_files(path_test, IJ, idx)
    append!(_configs,[coords2configs([IJ, R],Z,CylindricalBondEnvelope(rcut,renv,rcut/2),cell)])
    append!(_H,[H]); 
end


function rms(X::Vector{Matrix{Float64}}, Y::Vector{Matrix{Float64}}) 
    Z = map( (x,y) -> x .- y, X, Y )
    N = length(Z)*length(first(Z))
    return ( real(dot( Z, Z ))/N )^0.5
end

function rel(X::Vector{Matrix{Float64}}, Y::Vector{Matrix{Float64}}) 
    Z = map( (x,y) -> x./ (y .+ 0) .- 1, X, Y )
    N = length(Z)*length(first(Z))
    return ( real(dot( Z, Z ))/N )^0.5
end

function res(X::Vector{Matrix{Float64}}, Y::Vector{Matrix{Float64}})
    N = length(X)*length(first(X))
    residuals = map( (x,y) -> norm(x .- y)/N^0.5, X, Y )
    return residuals
end


rms_jig = test_setup(c, basis, rms)
rel_jig = test_setup(c, basis, rel)

println("train")
println( JSON.json(Dict( "rmse" => vec(rms_jig( _H[1],_configs[1])), "rel_err" => vec(rel_jig(_H[1],_configs[1])) )))

println("test")
println( JSON.json(Dict( "rmse" => vec(rms_jig(_H[2],_configs[2])), "rel_err" => vec(rel_jig(_H[2],_configs[2])) )))