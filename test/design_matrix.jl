using goofy
using LinearAlgebra, ACE
using ACE: CylindricalBondEnvelope, State, rand_rot
using Random
using ACE.Wigner: wigner_D

# denne testen regner ut en designmatrise ved bruk av funksjon
# design_matrix og sammenligner med en som er laget for hånd

function indicator(X) 
    m::Int64, n::Int64 = size(X)
    i::Int64 = 0; j::Int64 = 0
    Xtilde::Matrix{Float64} = zeros(Float64, m,n)
    for i in 1:m
        for j in 1:n
            Xtilde[i,j] = Int( abs(X[i,j]) > 1e-15 )
        end
    end
    return Xtilde    
end



using SparseArrays

function ctran(l::Int64,m::Int64,μ::Int64)
   if abs(m) ≠ abs(μ)
      return 0
   elseif abs(m) == 0
      return 1
   elseif m > 0 && μ > 0
      return 1/sqrt(2)
   elseif m > 0 && μ < 0
      return (-1)^m/sqrt(2)
   elseif m < 0 && μ > 0
      return  - im * (-1)^m/sqrt(2)
   else
      return im/sqrt(2)
   end
end

ctran(l::Int64) = sparse(Matrix{ComplexF64}([ ctran(l,m,μ) for m = -l:l, μ = -l:l ]))

function evaluateval_real(Aval)
   L1,L2 = size(Aval[1])
   L1 = Int((L1-1)/2)
   L2 = Int((L2-1)/2)
   C1 = ctran(L1)
   C2 = ctran(L2)
   return real([ C1 * Aval[i].val * C2' for i = 1:length(Aval)])
end



n = 1
rcut = 50.0 ; renv = rcut/2; order = 1; degree = 5
envelope = CylindricalBondEnvelope(rcut, renv, renv)
basis = offsite_generator( envelope, order, degree )

Z = [14, 14]
IJ = [ (1, 2) ]

lat = 5.0
cell = Matrix(lat*I, 3, 3)
a1, a2, a3 = [cell[:,x] for x in 1:size(cell,2)]
R = [ lat*rand(2,3) for i in 1:2 ]

map = Dict(0 => 1, 1 => 3)
configs = coords2configs([IJ, R], Z, envelope,  cell)

# regn ut en full blokk på starten. Se at den står på riktig sted i riktig rekkefølge
# se på et sted der det ikke skal være en blokk fordi for relevant L1, L2 er basisen for kort. Bekreft at det står nuller der 
# regn ut en full blokk på slutten. Sjekk at det står på riktig sted.
failures = 0
for (k, L1) in enumerate(0:1)
    for (l, L2) in enumerate(0:1)
        B = basis(L1, L2)
        test = design_matrix(B, configs)
        pos = 1
        for (z, config) in enumerate(configs) 
            ref = [item for item in ACE.evaluate(B, config)]
            ref = evaluateval_real(ref)

            m, n = size(ref[1])
    
            block = zeros(ComplexF64, m*n, length( ref ) )
            for (x, item) in enumerate(ref)
                col = zeros(ComplexF64, m*n)
                y = 1
                for j in 1:n
                    for i in 1:m
                        col[y] = item[i,j]
                        y += 1
                    end
                end
                block[:,x] = col
            end
            if norm( test[pos:(pos + m*n-1),:] - block )/length( block )^0.5 > 1e-15
                failures += 1
            end
            pos += m*n
        end
    end
end


println("pass ", failures == 0 )