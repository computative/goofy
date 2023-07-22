using goofy, JSON, HDF5, LinearAlgebra, Random, Plots,ACE
using ACE: CylindricalBondEnvelope

# I set the degree, maxorder, cutoff and regularization parameter
rcut = 5.0; renv = rcut/2; 
order = 1; degree = 5
stride = 1
lambda = 1e-20
k = 2^4


N::Int64 = 100
R::Vector{Matrix{Float64}} = [ rand(2,3) for i in 1:N ]
IJ::Vector{Tuple{Int64,Int64}} = [ (1,2) for i in 1:N]
cell = [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0]
Z = Int64.(14*ones(length(IJ))) # I assume Z is the same for every system

envelope::CylindricalBondEnvelope = ACE.CylindricalBondEnvelope(rcut, renv, renv)

H::Vector{Matrix{ComplexF64}} = []

configs = coords2configs([IJ, R], Z, envelope, cell)

for r in R#config in configs
    RIJ = vec(reduce(hcat, config.Xs[1].rr0.data))
    mat::Matrix{ComplexF64} = zeros(ComplexF64, 4, 4)
    mat[1,1] = dot(RIJ,RIJ)
    mat[1,2:4] = RIJ
    mat[2:4,1] = RIJ
    mat[2:4,2:4] = RIJ * RIJ'
    append!(H, [mat])
end


L_cfg = Dict(0=>1, 1=>1)
ace_param = [degree, order, rcut, renv, L_cfg]
fit_param = [H[1:stride:end], lambda]
system = [IJ[1:stride:end], R[1:stride:end], Z, cell]


coef, fitted, e, basis, configs = train(system, ace_param, fit_param)

test( coef, basis, configs, H[1:stride:end] )
    
