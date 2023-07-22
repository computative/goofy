using goofy, JSON, HDF5, LinearAlgebra, Random, Statistics
using ACE: BondEnvelope, CylindricalBondEnvelope

# learning curves from the terminal


# I set the degree, maxorder, cutoff and regularization parameter
rcut = 5.0; renv = rcut/2; 
order = 2; degree = 10
lambda = 1e-20

# it is assumed that the order of the orbitals in the sub-blocks are the following:
# first come all the s-orbitals, then all the p-orbitals, then the d-orbitals and so on

# basis config

H = []; R = []; IJ = []; unitcell = []
for id in ARGS[1:2]

    f = HDF5.h5open( "/home/marius/Dokumenter/Skole/phd/goofy.git/test/" * id * ".h5", "r")
    matrices = [HDF5.read( f, string(i) ) for i in 0:99]
    HDF5.close(f)

    raw = JSON.parsefile("/home/marius/Dokumenter/Skole/phd/goofy.git/test/" * id * ".json")
    m = length(raw); n = length(raw["1"]); l = length(raw["1"][1])
    coords = [zeros(n,l) for i in 1:m]

    for (key,value) in raw
        i = parse(Int64,key)
        for (j, vec) in enumerate(value)
            coords[i+1][j,:] = Float64.( vec )
        end
    end


    # for hver index (0-99) må vi velge ut noe som er nærmere enn cutoff
    # remember to pick some coordinates that have nonzero cutoff.
    # this is a lazy method but avoids the boundaries of the material

    k = parse(Int64, ARGS[3]); N = 4; _IJ::Vector{Tuple{Int64,Int64}} = []; 
    _H::Vector{Matrix{ComplexF64}} = []; _R::Vector{Matrix{Float64}} = []
    for (coord, matrix) in zip(coords, matrices)
    I::Int64 = 0; J::Int64 = 0
    choice = [] # this will contain coordinates of Hamiltonian blocks
    while length(choice) < k # while I am yet to collect k suitable coordinates
        I, J = randperm(size(coord)[1])[1:2] # I try distinct coords (that produces offsite blocks)
        if LinearAlgebra.norm(coord[I,:] - coord[J,:]) < rcut # Only if atoms are close enough ...
            append!(choice, [(I,J)] ) # ... their indices are added to the array of chosen coordinates
        end
    end
    for (I,J) in choice
        i = N*(I-1)+1
        j = N*(J-1)+1
        (N*(J-1)+1):( N*J )
        append!(_H, [ matrix[i:(i+N-1), j:(j+N-1)] ] ) # The hamiltonian-array is extended by our choices
    end
    append!(_IJ,choice)  # I extend the 'IJ'-array by the chosen coordinates
    append!(_R, [coord for i in 1:k]) # R contains the coordinates of the systems
    end

    # I assume that the cell does not change for each sample.
    _unitcell = eval(Meta.parse(read("/home/marius/Dokumenter/Skole/phd/goofy.git/test/" * id * ".cell", String)))

    append!(R,[_R]); append!(IJ,[_IJ]); append!(H,[_H]); append!(unitcell,[_unitcell])
end


Z = Int64.(14*ones(length(IJ[1]))) # I assume Z is the same for every system


# H, R are collections of configurations, Z and cell are charge and unit cell for all configs

L_cfg = Dict(0=>1, 1=>1)
ace_param = [degree, order, rcut, renv, L_cfg]
fit_param = [H[1], lambda]
system = [IJ[1], R[1], Z, unitcell[1]]


c, fitted, residuals, basis, configs = train(system, ace_param, fit_param)

rmse_train = test(c, basis, configs, H[1])
rmse_test = test(c, basis, coords2configs([IJ[2], R[2]], Z, CylindricalBondEnvelope(rcut, renv, renv), unitcell[2]), H[2])

println(rmse_train)
println(rmse_test)



