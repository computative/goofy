using goofy, LinearAlgebra, ACE
using ACE: CylindricalBondEnvelope, State, rand_rot
using Random
using ACE.Wigner: wigner_D

# denne testen roterer koordinater enten manuelt, eller multipliserer med
# rotasjonsmatrise for å sjekke at det medfører samme resultat 

n = 1; N = 5 # n : num systems and N : number of atons
rcut = 5.0 ; renv = rcut/2; order = 1; degree = 5
envelope = CylindricalBondEnvelope(rcut, renv, renv)
basis = offsite_generator( envelope, order, degree )

L1, L2 = rand((0,1), 2)
B = basis(L1, L2)

Z = Int64.(14*ones(N))
IJ = [ Tuple(randperm(N)[1:2])  for i in 1:n ]
lat = 5.0
cell = Matrix(lat*I, 3, 3)
a1, a2, a3 = [cell[:,x] for x in 1:size(cell,2)]
R = [lat*rand(N,3) for i in 1:n]



# random rotation and wigner
Q = rand_rot()
D1 = wigner_D(L1, Q)
D2 = wigner_D(L2, Q)

ref_config = coords2configs([IJ, R], Z, envelope,  cell)[1]
ref = ACE.evaluate(B, ref_config)

test_config = ACEConfig( [State(rr = Q * ref_config.Xs[i].rr, rr0 = Q * ref_config.Xs[i].rr0, be = ref_config.Xs[i].be) for i = 1:length(ref_config)] )
test = ACE.evaluate(B, test_config)

m, n = size(test[1].val)
C = Ylm_complex2real(m,n)

D = [vec((D1) * ref[x].val * (D2')) for x in 1:length(ref)]
F = [vec(test[x].val) for x in 1:length(test)]

println("pass " , norm(D-F) < 1e-10)

A = [0 1 0;0 0 1; -1 0 0];

println( (Q') * A' * C[1] * ref[18].val * (C[2]') * A * (Q) )
println(        A' * C[1] * test[18].val * (C[2]') * A ) 

# jeg skjønner ikke hvorfor denne ikke transformerer ordentlig