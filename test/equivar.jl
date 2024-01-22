using goofy, LinearAlgebra, ACE
using ACE: CylindricalBondEnvelope, State, rand_rot
using Random
using ACE.Wigner: wigner_D

# denne testen sjekker at basisen har riktig rotasjonsegenskap
# ved at den baade transformerer koordinatene med matrisen Q 
# og transformerer operatoren med wignermatrisene

pass = true

for i in 1:20
    n = rand(1:10); N = rand(2:10) # n : num systems and N : number of atoms
    rcut = 5.0 ; renv = rcut/2; order = 1; degree = 5
    envelope = CylindricalBondEnvelope(rcut, renv, renv)
    basis = offsite_generator( envelope, order, degree )

    Z = [ Int64.(14*ones(N)) for i in 1:n ]
    IJ = [ Tuple(randperm(N)[1:2]) for i in 1:n ]
    cell = [1.0*Matrix(I, 3, 3) for i in 1:n ]
    R = [rand(3,N) for i in 1:n]

    # random rotation and wigner
    L1, L2 = rand((0,1), 2)
    Q = rand_rot()
    D1, D2 = wigner_D(L1, Q), wigner_D(L2, Q)

    # a pair of sets of basis-matrices
    configs = coords2configs([IJ, R], Z, envelope,  cell)
    Qconfigs = coords2configs([IJ, [Q * R[i] for i in eachindex(R)]], Z, envelope,  [Q * cell[i] for i in eachindex(R)] )
    for (config, Qconfig) in zip(configs,Qconfigs)
        B = ACE.evaluate(basis(L1, L2), config)
        QB = ACE.evaluate(basis(L1, L2), Qconfig)
        global pass
        pass &= isapprox([vec((D1) * B[x].val * (D2')) for x in 1:length(B)], 
                         [vec(QB[x].val) for x in 1:length(QB)])
    end
end
@show pass