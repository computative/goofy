using goofy, LinearAlgebra, ACE, Random, JSON
using ACE: CylindricalBondEnvelope, State, rand_rot
using ACE.Wigner: wigner_D


function python(output)
    println( JSON.json(output) )
end


input = JSON.parse(ARGS[1])

# random rotation and wigner
Q = rand_rot()
D1, D2 = wigner_D(input["L1"], Q), wigner_D(input["L2"], Q)


D = Dict( "Q" => Q', "D1" => D1', "D2" => D2' )

python(D)



