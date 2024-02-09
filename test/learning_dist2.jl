using goofy, JSON, HDF5, LinearAlgebra, Random, Dates
using ACE: BondEnvelope, CylindricalBondEnvelope

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
    for _ in ["train", "test"]
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


    c, _, _, basis, configs = train(system, ace_param, fit_param)
    
    label = join(split(split(path, "/")[end-1], "_")[2:end-1], "-")
    timestamp   = Dates.format(now(), "dd-mm-yyTHH-MM-SS")
    outpath = abspath(@__DIR__, "../../goofy.files/models/L")
    write_item((basis,c),outpath * string(len) * "-" *  label * "-" * timestamp * ".mdl")
    write_item(chosen[1],outpath * string(len) * "-" *  label * "-" * timestamp * ".cho")

    # testing
    
    test_configs = coords2configs([_IJ[2], _R[2]], _Z[2], 
                        CylindricalBondEnvelope(rcut, renv, rcut/2), _unitcell[2])
    
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
    

    res_jig = test_setup(c, basis, res)

    settings = Dict(
        "length" => len, 
        "degree" => degree,
        "order" => order,
        "rcut" => rcut,
        "lambda" => lambda
    )

    symm = ["SS","PS","SP","PP"]
    cfg = [configs, test_configs]

    for (k, mode) in enumerate(["train", "test"])       
        res_dict = Dict( symm .=> vec(res_jig(_H[k], cfg[k] ) ) )
        sizes = map(j -> [ abs(_H[k][i][j]) for i in eachindex(_H[k]) ], collect(eachindex(basis)))
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


end




