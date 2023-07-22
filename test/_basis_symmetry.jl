using goofy, LinearAlgebra, Random
using ACE: BondEnvelope, CylindricalBondEnvelope, rand_rot
using ACE.Wigner: wigner_D


n = 1 # how many systems do you want to test for?
N = 5 # how many particles in a system?
rcut = 5.0 ; renv = rcut/2
envelope::CylindricalBondEnvelope = CylindricalBondEnvelope(rcut, renv, renv)
lat::Float64 = 5
cell = Matrix(lat*I, 3, 3)
a1,a2,a3 = [cell[:,x] for x in 1:size(cell,2)]
R = [lat*rand(N,3) for i in 1:n]
Z = Int64.(14*ones(N))
IJ::Vector{Tuple{Int64,Int64}} = [Tuple(randperm(5)[1:2]) for i in 1:n]


coords = [IJ, R]




ref_config = coords2configs([IJ, R], Z, envelope,  cell)[1]
ref = ACE.evaluate(B, ref_config)

# random rotation and wigner
Q = rand_rot()

test_config = ACEConfig( [State(rr = Q * ref_config.Xs[i].rr, rr0 = Q * ref_config.Xs[i].rr0, be = ref_config.Xs[i].be) for i = 1:length(ref_config)] )
test = ACE.evaluate(B, test_config)

L1 = 1; L2 = 1
D1, D2 = wigner_D(L1, Q), wigner_D(L2, Q)

D = [vec((D1')*test[x].val*D2) for x in 1:length(test)]
F = [vec(ref[x].val) for x in 1:length(ref)]
