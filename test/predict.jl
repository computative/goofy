using goofy, JSON, HDF5, LinearAlgebra, Random, Statistics
using ACE: BondEnvelope, CylindricalBondEnvelope
using Dates

# execute with julia --project=. test/learning.jl r l p p p p
# rcut len path_test path_train path_choice path_model

rcut = parse(Float64, ARGS[1])
renv = rcut/2; 
len = parse(Int64, ARGS[2])
order = 6
degree = 5
lambda = 1e-11

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
    append!(_configs,[coords2configs(IJ, R,Z,cell,CylindricalBondEnvelope(rcut,renv,rcut/2))])
    append!(_H,[H]); 
end


function res(X::Vector{Matrix{Float64}}, Y::Vector{Matrix{Float64}})
    N = length(X)*length(first(X))
    residuals = map( (x,y) -> norm(x .- y)/N^0.5, X, Y )
    return residuals
end

res_jig = test_setup(c, basis, res)

settings = Dict(
    "length" => len, 
    "degree" => degree,
    "order" => order,
    "rcut" => rcut,
    "lambda" => lambda
)

symm = ["SS","PS","SP","PP"]
cfg = [_configs[1], _configs[2]]

slices = [ (1,1)  (1,2:4); (2:4,1) (2:4,2:4) ]

for (k, mode) in enumerate(["train", "test"])       
    res_dict = Dict( symm .=> vec(res_jig(_H[k], cfg[k] ) ) )
    # i er hver matrise, j er hvert idx i en block
    sizes = map(j -> [ mean(abs.(_H[k][i][slices[j]...])) for i in eachindex(_H[k]) ], collect(eachindex(basis)))
    size_dict = Dict( symm .=> sizes )
    
    d = Dict(
        "mode" => mode,
        "size" => size_dict, 
        "residuals" => res_dict, 
        "separation" =>  [norm(first(cfg[k][i]).rr0 ) for i in eachindex(cfg[k]) ]
    )
    print("d = ")
    println(JSON.json(merge(settings, d)))
        
end