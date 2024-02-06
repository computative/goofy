using ACE:rand_rot
using ACE.Wigner: wigner_D

Q = rand_rot()
D1, D2 = wigner_D(0,Q), wigner_D(1,Q)

@show(Q)
@show(D1)
@show(D2)