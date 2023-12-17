using goofy, JSON, HDF5, LinearAlgebra, ACE, Random, Statistics, Plots
using ACE: BondEnvelope, CylindricalBondEnvelope, evaluate

# learning curves from the terminal
# execute with julia --project=. test/learning.jl n m l u v  (where n and m are the id of the training and testing data. 
# l is regularization parameter, u and v are the order and degree respectively


# basis config
# I set the degree, maxorder, cutoff and regularization parameter
rcut = 5.0; renv = rcut/2; 
path = "/home/marius/Dokumenter/Skole/phd/goofy.git/test"

file_ids = 1 #ARGS[1:2]
lambda = parse(Float64, ARGS[3])
order = parse(Int64, ARGS[4]); degree = parse(Int64, ARGS[5])
#lens = parse.(Int, split(chop(ARGS[6]; head=1, tail=1), ','))
target_matrix_id = "0" # which matrix index in file id 1 should be the target for the prediction?


# it is assumed that the order of the orbitals in the sub-blocks are the following:
# first come all the s-orbitals, then all the p-orbitals, then the d-orbitals and so on



f = HDF5.h5open( path * "/1.h5", "r")
_H = HDF5.read( f, target_matrix_id )
HDF5.close(f)

H, R, IJ, cell, Z = parse_files(path, string(1), 10,1)

mu::Int64, nu::Int64 = size(H[1]) # size of sub_block
m::Int64,n::Int64 = size(_H)
p::Int64,q::Int64 = (2,2) # size of basis
M::Int64 = div(m,mu)
N::Int64 = div(n,nu)




raw = JSON.parsefile(path * "/1.json")
_R = transpose([ raw[ target_matrix_id ][i][j] for j in 1:3 , i in 1:length( raw[ target_matrix_id ] )  ])
_Rp = [ _R for i in 1:(M*N)  ]
_IJ = reshape([ (I,J) for  J in 1:N, I in 1:M ], M, N)
_IJ[diagind(_IJ)] = [(1,2) for _ in 1:M]
_IJ = reshape(_IJ, M*N)
_coords = [_IJ, _Rp]
_Z = [ Z[1] for i in 1:(M*N)  ]
_cell = [cell[1] for i in 1:(M*N) ]
configs = coords2configs(_coords, _Z, CylindricalBondEnvelope(norm(diag(cell[1])), renv, rcut/2), _cell)
rel_dist = reshape(norm.([ config.Xs[1].rr0 for config in configs ]), M, N)
rel_dist[diagind(rel_dist)] = repeat([0],N)

mask = (0.0 .< reshape(rel_dist,M*N) .< rcut)
_IJ = [ _IJ[i] for i in 1:(M*N) if mask[i] == 1 ]
_R = [ _R for i in 1:(M*N) if mask[i] == 1 ]
_Z = [ Z[1] for i in 1:(M*N) if mask[i] == 1 ]
_cell = [cell[1] for i in 1:(M*N) if mask[i] == 1 ]
_coords = [_IJ, _R]
configs = coords2configs(_coords, _Z, CylindricalBondEnvelope(rcut, renv, rcut/2), _cell) 

for _ in 1:100 
    
    H, R, IJ, cell, Z = parse_files(path, string(1), 100,1)
    L_cfg = Dict(0=>1, 1=>1)  # r0cut rcut
    ace_param = [degree, order, rcut, renv, L_cfg]
    fit_param = [H, lambda, "none"]
    system = [IJ, R, Z, cell]
    coef, _, residuals, basis, _ = train(system, ace_param, fit_param)

    Hpredict = zeros(ComplexF64, n, m)
    v::Int64 = 1
    w::Int64 = 1
    A::Int64 = 1
    b::Int64 = 1
    res::ComplexF64 = 1.0
    block::Matrix{ComplexF64} = zeros(mu, nu)
    position_mask = 0
    position_cfg = 0
    for I in 1:M
        for J in 1:N 
            position_mask += 1
            if mask[position_mask] == 0
                continue
            else
                position_cfg += 1
                config = configs[position_cfg] #avhenger av IJ
            end
            # are there a diagonal elt?
            block = zeros(mu,nu)
            A = 1
            for a in 1:p
                B = 1
                for b in 1:q
                    v = 2*a - 1; w = 2*b - 1  # size that symmetry type will occupy in block
                    C::Vector{Matrix{ComplexF64}} = Ylm_complex2real(v,w)                                          
                    D = real([ C[1] * Matrix(item.val) * C[2]' for item in ACE.evaluate(basis[a,b], config )  ] )  
                    block[A:(A+v-1),B:(B+w-1)] =  sum([ D[i] * coef[a,b][i] for i in 1:length(coef[a,b]) ])        
                    B += w
                end
                A += v
            end
            Hpredict[( 4*(I-1) + 1):( 4*I ), ( 4*(J-1) + 1):( 4*J ) ] = block
        end
    end
    for I in 1:N
        Hpredict[( 4*(I-1) + 1):( 4*I ), ( 4*(I-1) + 1):( 4*I ) ] = _H[( 4*(I-1) + 1):( 4*I ), ( 4*(I-1) + 1):( 4*I ) ]
    end
    # sett inn diagonal elts


    Eexact = real(eigvals(_H))
    Epredict = real(eigvals(Hpredict))

    println("Eexact ", Eexact)
    println("Epredict ", Epredict)
end


