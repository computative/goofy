using ACE, StaticArrays
using ACE: Utils.RnYlm_1pbasis, SphericalMatrix, SymmetricBasis, State, ACEConfig
using JuLIP, LinearAlgebra

# Offsite Basis construction
## TODO: if you need onsite, you will need to construct a onsite basis here

r0 = 2.5; rin = 0.5 * r0; rcut = 10.0; renv = 10-rcut/2
env = ACE.CylindricalBondEnvelope(rcut,renv,renv)

# this should give 8 symmetric basis elements, because consider all possibilities. Use the notation v = (n,l,m)
ord = 1; maxdeg = 4 
# we have (1,0,0), (2,0,0), (2,1,m), (3,1,m); with 3 choices of m for each of the v (m = -1,0,1). As you can see,
# these are 8 possibilities.
Bsel = SimpleSparseBasis(ord, maxdeg)

# oneparticle basis
RnYlm = RnYlm_1pbasis(; maxdeg = maxdeg, r0 = r0, trans = PolyTransform(1, 1/r0^2), rcut = sqrt(renv^2+(rcut+renv)^2), rin = rin, pcut = 2)
B1p_env = env * RnYlm
B1p = ACE.Categorical1pBasis([:bond, :env]; varsym = :be, idxsym=:be) * B1p_env


function basegen(L1,L2)
    return SymmetricBasis(SphericalMatrix(L1, L2; T = ComplexF64), B1p, Bsel)
 end


N = 3; configs = []
for i in 1:N
    rr0 = SVector{3,Float64}(rand(3))
    R_IJ = ACE.State(rr = SVector{3,Float64}(rand(3)), rr0 = rr0, be = :bond) 
    R_IK = ACE.State(rr = SVector{3,Float64}(rand(3)), rr0 = rr0, be = :env) 
    config = ACEConfig([R_IJ,R_IK])
    append!(configs,[config])
end


############################################


a = 1
b = 1

function columnize(mat::Matrix{T} where T <: Union{Float64, ComplexF64})
    m,n = size(mat) 
    return reshape(mat, m*n)
end



function uncolumnize(v::Vector{T} where T <: Union{Float64, ComplexF64}, m::Int64, n::Int64 )
    return reshape(v, (m,n) )
end


function Ylm_complex2real(m::Int64,n::Int64)
    L1 = (m-1)/2; L2 = (n-1)/2; 
    C::Vector{Matrix{ComplexF64}} = [zeros(ComplexF64, m,m), zeros(ComplexF64, n,n)]
    for (a,l) in enumerate([L1,L2])
       for (b,nu) in enumerate(Vector(-l:l)) 
          for (c,mu) in enumerate(Vector(-l:l))
             if abs(nu) != abs(mu)
                C[a][b,c] = 0
             elseif abs(nu) == 0
                C[a][b,c] = 1
             elseif nu > 0 && mu > 0
                C[a][b,c] = 1/sqrt(2)
             elseif nu > 0 && mu < 0
                C[a][b,c] = (-1)^m/sqrt(2)
             elseif nu < 0 && mu > 0
                C[a][b,c] = - im * (-1)^nu/sqrt(2)
             else
                C[a][b,c] = im/sqrt(2)
             end
          end
       end
    end
    return C
end

function eval(basis, config)
    [ C[1] * Matrix(item.val) * C[2]' for item in ACE.evaluate(basis, config )  ]
    return 0
end



L_cfg = Dict(0=>1, 1=>1)

# basegen is now a function that returns a basis when you give it a pair (L1,L2)
basize = sum(collect(values(L_cfg))) # basis size (in tight binding-sense): Total orbital count
c::Matrix{Vector{ComplexF64}} = [ [0.0] for _ in 1:basize, _ in 1:basize] # this will hold coeffs
fitted::Matrix{Vector{ComplexF64}} = [ [0.0] for _ in 1:basize, _ in 1:basize] # fitted values
e::Matrix{Vector{ComplexF64}} = [ [0.0] for _ in 1:basize, _ in 1:basize] # residuals
L_count = sum(collect(values(L_cfg))) # the total sum of 'counts of each orbital symmetry-type'
L = sort(collect(keys(L_cfg))) # the keys are the user-selected values for L1 and L2.
basis::Array{SymmetricBasis, 2} = Array{SymmetricBasis}(undef, L_count, L_count)
a::Int64 = 1; A::Int64 = 1
for L1 in L, _ in 1:L_cfg[L1]
   m::Int64 = 0; n::Int64 = 0
   b::Int64 = 1; B::Int64 = 1
   for L2 in L, _ in 1:L_cfg[L2]
      basis[a,b] = basegen(L1, L2)
      m , n = size(ACE.evaluate(basis[a,b], configs[1])[1].val)
      Y = zeros( ComplexF64, length(H)*m*n )
      for (c,block) in enumerate(H)
         subblock = block[A:(A+m-1), B:(B+n-1)]
         col = zeros(ComplexF64, m*n)
         d = 1
         for v in 1:size(subblock)[2]
            for u in 1:size(subblock)[1]
               col[d] = subblock[u,v]
               d += 1
            end
         end
         Y[  ((c-1)*length(col) + 1): c*length(col) ] = col
      end


      X = design_matrix(basis[a,b], configs) # make design-matrices by evaluating the basis at the configs
      #if (L1 == 1 && L2 == 0) || (L1 == 0 && L2 == 1)
      #   println(X)
      #end
      Xt = transpose(X) # I will need the regularization parameter and X transpose
      c[a,b] = vec( (Xt * X + lambda * I) \ (Xt * Y) )  # these are the coefficients
      fitted[a,b] = X*c[a,b] # these are the fitted values ...
      e[a,b] = Y - fitted[a,b] # ... and the residuals
      b += 1; B += n
   end
   a += 1; A += m
end




## Evaluation
#B = ACE.evaluate(basis,config)
#
## Model
#c = rand(length(basis))
#SModel = ACE.LinearACEModel(basis,c)
#ACE.evaluate(SModel, Config)
#
#sum([B[i] * c[i] for i = 1:length(c)]) == ACE.evaluate(SModel, Config)





