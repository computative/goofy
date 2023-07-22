using goofy
using LinearAlgebra, ACE
using ACE: CylindricalBondEnvelope, State, rand_rot
using Random
using ACE.Wigner: wigner_D


n = 1
rcut = 50.0 ; renv = rcut/2; order = 1; degree = 5
envelope = CylindricalBondEnvelope(rcut, renv, renv)
basis = offsite_generator( envelope, order, degree )

Z = [14,14]
IJ = [ (1,2) ]

lat = 50.0
cell = Matrix(lat*I, 3, 3)
a1, a2, a3 = [cell[:,x] for x in 1:size(cell,2)]
R = [[ 0.0 0.0 0.0 ; 45.0 0.0 0.0]]

map = Dict(0 => 1, 1 => 3)
block = zeros(ComplexF64, 4, 4)
config = coords2configs([IJ, R], Z, envelope,  cell)[1]
k = 1
for L1 in 0:1
    l = 1
    for L2 in 0:1
        B = basis(L1, L2)
        res = ACE.evaluate(B, config)
        
        #println(block[k:k+map[L1]-1,l:l+map[L2]-1]) 
        #println(res[1].val)
        block[k:k+map[L1]-1,l:l+map[L2]-1] = res[1].val
        l += map[L2]
    end
    k += map[L1]
end
    
block 