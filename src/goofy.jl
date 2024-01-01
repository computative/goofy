module goofy

export coords2configs, offsite_generator, design_matrix, train, predict, test_setup, AA, Ylm_complex2real, parse_files, random_idx, parse_files_depricated, depricated_test, qr_solver, lsqr_solver, inv_solver

using JSON, HDF5, JuLIP, Statistics, Plots, Printf, LinearAlgebra, InteractiveUtils, Random, IterativeSolvers
using StaticArrays, LowRankApprox, IterativeSolvers, Distributed, DistributedArrays, ACE
import ACE.scaling, ACE.write_dict, ACE.read_dict
using ACE: PolyTransform, SphericalMatrix, PIBasis, SymmetricBasis,
           SimpleSparseBasis, Utils.RnYlm_1pbasis,  
           Categorical1pBasis, filter, State, ACEConfig,
           evaluate_d, get_spec, evaluate, PositionState, BondEnvelope, CylindricalBondEnvelope



# n should divide the number of hamiltonians in the datafile
# path : h5-file. n : number of observations, rcut : cutoff radius, 
# natoms : number of atoms in system, chosen: a matrix whose columns are (IJ, index) 
function random_idx(path::String, n::Int64, rcut::Real)
   infile::HDF5.File = HDF5.h5open(path)
   num_ham_in_datafile::Int64 = length(infile["ham"])
   num_obs_per_ham::Int64 = Int64(ceil(n/num_ham_in_datafile))
   R::Matrix{Float64} = zeros(2,2)
   chosen = []
   for ham_id in 1:num_ham_in_datafile
      R = infile["pos"][string(ham_id)][:,:]
      while (length(chosen) < ham_id*num_obs_per_ham) & (length(chosen) < n)
         I, J = randperm(size(R)[2])[1:2]
         if norm(R[:,J] - R[:,I]) < rcut # choose only atoms closer than rcut
            append!(chosen, [[(I,J), ham_id]] )
         end
      end
   end
   HDF5.close(infile)
   # I convert a vector of vectors to matrix and return that matrix
   return permutedims(reduce(hcat , chosen)) #[ chosen[i][j] for i in eachindex(chosen), j in 1:2]
end


# this function takes a vector of chosen indecies and the number of atoms in the system 
# and returns finished parsed data 
function parse_files(path, IJ, idx )
   infile = HDF5.h5open(path)
   num_atoms = size(first(infile["pos"]))[2]
   block_size = Int64(round(size(first(infile["ham"])[:,:])[1]/num_atoms))

   sl1 = [ ((I-1)*block_size+1):(I*block_size) for (I,_) in IJ ]
   sl2 = [ ((J-1)*block_size+1):(J*block_size) for (_,J) in IJ ]

   # the hdf_select-function selects blocks of matrices with given idx  and group g
   select_vec = (file, g, idx, slice) -> file[g][string(idx)][slice]
   select_mat = (file, g, idx, sl1, sl2) -> file[g][string(idx)][sl2, sl1] # transpose!
   
   ham = map( (i,j,k) -> permutedims(select_mat(infile, "ham", k, i, j)), sl1, sl2, idx)
   pos = map( k -> select_mat(infile, "pos",   k,  :, :), idx )
   cell = map( k -> select_mat(infile, "cell",  k,  :, :), idx )
   species = map( k -> select_vec(infile, "species", k, :), idx )
   
   HDF5.close(infile)
   return ham, pos, cell, species
end


function coords2configs(coords, Z, envelope,  cell)
   IJ::Vector{Tuple{Int64, Int64}}, R::Vector{Matrix{Float64}} = coords
   # -!- DET BLIR VIKTIG SJEKKE AT INGEN FLERE URPESISHETER Á LA rcut vs Rcut er tilstede i hello world jeg fikk fra liwei
   r0cut::Float64 = envelope.r0cut; zcut::Float64 = envelope.zcut; rcut::Float64 = envelope.rcut
   configs = [] # this array will contain all the configurations (ACEConfig)
   for (i,(I,J)) in enumerate(IJ) # this loops runs over the coords of atoms that forms a bond
      # JuLIP can output rel. coords. It requires an atoms "object" and uses Potentials.neigsz
      at = JuLIP.Atoms(; X = R[i], Z = Z[i], cell = cell[i]', pbc = [true, true, true])
      # here I make a large system that is repeated according to pbc
      Rcut = ((r0cut + rcut)^2 + zcut^2)^0.5
      neigh_I = JuLIP.Potentials.neigsz(JuLIP.neighbourlist(at, Rcut), at, I)
      # the large system contains repeated images of the same vectors. The following selects the
      # index of the image of the vector R_J that is closest to R_I and calls it idx
      idx = findall(isequal(J),neigh_I[1])
      idx = idx[findmin(norm.(neigh_I[2][idx]))[2]]
      R_IJ = neigh_I[2][idx] # this is the bond R_IJ that correspond to one of the H-block-matrices
      states = [State(rr = R_IJ, rr0 = R_IJ, be=:bond)] # a bond state is required in the config
      for R_IK in neigh_I[2][1:end .!= idx] # this is a loop over all coordinates different from R_IJ
         env_state = State(rr = R_IK, rr0 = R_IJ, be=:env) # for each R_IK, a state is made
         if filter(envelope, env_state) # if the state R_IK is inside the envelope...
            append!( states, [env_state] ) # ... then append it to the states-array
         end
      end 
      append!(configs, [ACEConfig(states)]) # states are can be cast to ACEConfig and added to configs
   end
   return configs
end



function filter_offsite_be(bb,maxdeg,λ_n=.5,λ_l=.5)
   if length(bb) == 0; return false; end
   deg_n = ceil(Int64,maxdeg * λ_n)
   deg_l = ceil(Int64,maxdeg * λ_l)
   for b in bb
      if (b.be == :env) && (b.n>deg_n || b.l>deg_l)
         return false
      end
   end
   return ( sum( b.be == :bond for b in bb ) == 1 )
end


function maxdeg2filterfun(maxdeg,λ_n=.5,λ_l=.5)
   return bb -> filter_offsite_be(bb,maxdeg,λ_n,λ_l)
end




#  num_L1 and num_L2 are integers that say how many orbitals of each type is used
function offsite_generator(env, order, maxdeg)
   Bsel = SimpleSparseBasis(order, maxdeg) # this sets the complexity of the basis
   # the following makes a 1 particle basis
   r0::Float64 = 2.5; rin::Float64 = 0.5 * r0
   RnYlm = RnYlm_1pbasis(; maxdeg = maxdeg, r0 = r0, trans = PolyTransform(1, 1/r0^2), 
            rcut = sqrt(env.zcut^2+(env.r0cut+env.rcut)^2), rin = rin, pcut = 2) 
   # here I make the 1p basis that ignores configurations outside BondEnvelope
   B1p_env = env * RnYlm # eq 20 (multiply by indicator func)
   B1p = ACE.Categorical1pBasis([:bond, :env]; varsym = :be, idxsym=:be) * B1p_env # sum of eqn 20
   # that is all which is needed to get the SymmetricBasis (script-B in the paper)
   function basis(L1,L2)
      return SymmetricBasis(SphericalMatrix(L1, L2; T = ComplexF64), B1p, Bsel)#; filterfun = maxdeg2filterfun(maxdeg,0.5,0.5))
   end
   return basis
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



function design_matrix(basis, configs, intercept=false)
   # l is an integer equal to the highest dimension of the span of the summetric basis. 
   l::Int64 = length(configs); m::Int64, n::Int64 = size(ACE.evaluate(basis, configs[1])[1].val)
   k::Int64 = length( [ Matrix(item.val) for item in ACE.evaluate(basis, configs[1]) ] )
   # memory allocation for the design matrix. Notice it has 3 dimensions (1 too many) to begin with.
   C = Ylm_complex2real(m,n)
   elts::Vector{Vector{Matrix{ComplexF64}}} = [[ zeros(m,n) for i in 1:k] for j in 1:l]
   for (i, config) in enumerate(configs) # for each configuration I will evaluate it under each SymmetricBasis
      elts[i] = real([ C[1] * Matrix(item.val) * C[2]' for item in ACE.evaluate(basis, config )  ])
   end
   elts = [ [ elts[i][j] for i in 1:l ] for j in 1:k ]
   X = reduce(hcat,[reduce(vcat,[reshape(elts[i][j],m*n) for j in 1:l]) for i in 1:k])
   if intercept
      _m, _n = size(X)
      X = hcat(ones(_m), X)
   end
   return X
end



# solvers accept a design matrix, response-vector and regularization-scalar 
# (in that order) and returns a vector of numbers that equal the coefficents.
function qr_solver(X::Matrix{T}, Y::Vector{T},lambda::Real) where T <: Number
   m::Int64, n::Int64 = size(X)
   U = vcat(X,lambda*I(n))
   V = vcat(Y,zeros(n))
   return qr(U) \ V
end

function lsqr_solver(X::Matrix{T}, Y::Vector{T},lambda::Real) where T <: Number
   return lsqr(X,Y, damp=lambda)
end

function inv_solver(X::Matrix{T}, Y::Vector{T},lambda::Real) where T <: Number
   Xt = X'
   M::Matrix{Number} = (Xt * X + lambda * I)
   @show cond(M)
   return vec( M \ (Xt * Y) )
end



# this function takes all the samples of the system and parameters. It returns a model struct
function train(system, ace_param, fit_param)
   degree::Int64, order::Int64, r0cut::Float64, rcut::Float64, L_cfg::Dict{Int64, Int64} = ace_param #
   H::Vector{Matrix{ComplexF64}}, lambda::Float64, method, intercept::Bool = fit_param #
   #H::Vector{Matrix{ComplexF64}}, lambda::Float64, method::String, intercept::Bool = fit_param #
   IJ, R::Vector{Matrix{Float64}}, Z::Vector{Vector{Int64}}, cell::Vector{Matrix{Float64}} = system # jeg har sjekket de tre linjene at riktig ting pakkes inn og ut på riktig sted
   # I can convert coords -> rel. coords -> configurations using the ingredients below
   envelope::CylindricalBondEnvelope = ACE.CylindricalBondEnvelope(r0cut, rcut, r0cut/2)

   # coords2configs : takes absolute coordinates and (1) makes relative coordinates and (2) output configs 
   configs = coords2configs([IJ, R], Z, envelope, cell) # rcut kan du få fra BondEnvelope 
   # order is the polynomial order, degree is d_max, and the Dict says how many S, P, D, ... 
   # orbitals that I want. Keys are integers: 0 refer to S orbital, 1 refer to P orbital and so on.
   # The values are an instruction of how many I want of each orbital-symmetry in the basis 
   # basegen is now a function that returns a basis when you give it a pair (L1,L2)
   basegen = offsite_generator( envelope, order, degree )

   basize = sum(collect(values(L_cfg))) # basis size (in tight binding-sense): Total orbital count
   coef::Matrix{Vector{ComplexF64}} = [ [0.0] for _ in 1:basize, _ in 1:basize] # this will hold coeffs
   fitted::Matrix{Vector{ComplexF64}} = [ [0.0] for _ in 1:basize, _ in 1:basize] # fitted values
   residuals::Matrix{Vector{ComplexF64}} = [ [0.0] for _ in 1:basize, _ in 1:basize] # residuals
   L_count = sum(collect(values(L_cfg))) # the total sum of 'counts of each orbital symmetry-type'
   L = sort(collect(keys(L_cfg))) # the keys are the user-selected values for L1 and L2.
   basis::Array{SymmetricBasis, 2} = Array{SymmetricBasis}(undef, L_count, L_count)
   a::Int64 = 1; A::Int64 = 1
   # these are loops over all L1,L2 and over repeated orbital types
   for L1 in L, _ in 1:L_cfg[L1]
      m::Int64 = 0; n::Int64 = 0
      b::Int64 = 1; B::Int64 = 1
      for L2 in L, _ in 1:L_cfg[L2]
         basis[a,b] = basegen(L1, L2)
         m , n = size(ACE.evaluate(basis[a,b], configs[1])[1].val)
         Y = zeros( ComplexF64, length(H)*m*n )
         for (c, block) in enumerate(H)
            Y[  ( (c-1)*m*n +1 ): ( c*m*n ) ] = reshape(block[A:(A+m-1), B:(B+n-1)], m*n )
         end
         # this makes a design matrix ( symmetric basis, vector of configs) -> Regular matrix with samples 
         #                                                                          along axis 1 and basis along axis 2 
         X = design_matrix(basis[a,b], configs, intercept) # make design-matrices by evaluating the basis at the configs
         coef[a,b] = method(X, Y, lambda)
         fitted[a,b] = X*coef[a,b] # these are the fitted values ...
         residuals[a,b] = Y - fitted[a,b] # ... and the residuals
         b += 1; B += n
      end
      a += 1; A += m
   end
   return coef, fitted, residuals, basis, configs
end

function predict(coef::Matrix{Vector{T}}, basis::Array{SymmetricBasis, 2}, 
                                       configs, intercept=false) where T <: Number

   m::Int64 , n::Int64 = size(basis) # antall symmetrityper. I vårt tilfelle 2x2 
   K = [ size( first( ACE.evaluate(basis[v,w], first(configs)) ).val ) for v in 1:m , w in 1:n ]
   k::Int64 = sum([ item[1] for item in K[:,1]]) # størrelse på hver H-matrise (e.g. 4x4)
   l::Int64 = length(configs); # ant obervasjoner
   Hpredict::Array{ComplexF64, 3} = zeros(l, k, k)
   A::Int64 = 1; v::Int64 = 1; w::Int64 = 1
   for a in 1:m
      B::Int64 = 1
      for b in 1:n
         v,w = size( first(ACE.evaluate(basis[a,b], first(configs))).val) # dimensjonene til blokken (i,j) 
         C::Vector{Matrix{ComplexF64}} = Ylm_complex2real(v,w)
         res::Vector{Matrix{ComplexF64}} = [ zeros(ComplexF64, v,w) for i in 1:l ]
         for (c, config) in enumerate(configs) # her er det feil. a,b må være strl ikke L1,L2
            D = real([ C[1] * Matrix(item.val) * C[2]' for item in ACE.evaluate(basis[a,b], config )  ] )
            if intercept
               _m,_n = size(D[1])
               D0 = ones(_m,_n)
               res[c] = D0 * coef[a,b][1] + sum([ D[i-1] * coef[a,b][i] for i in 2:length(coef[a,b]) ])
            else
               res[c] = sum([ D[i] * coef[a,b][i] for i in 1:length(coef[a,b]) ])
            end
         end
         Hpredict[:,A:(A+v-1),B:(B+w-1)] =  [res[r][s,t] for r in 1:l, s in 1:v, t in 1:w ]
         B += w
      end
      A += v
   end
   return [ [Hpredict[a,b,c]  for b in 1:k , c in 1:k] for a in 1:l ]
end



function test_setup(coef::Matrix{Vector{T}}, basis::Array{SymmetricBasis, 2}, 
                                             method, intercept = false) where T <: Number
   function test_jig( H, configs)
      Hpredict::Vector{Matrix{ComplexF64}} = predict(coef, basis, configs, intercept)
      #E::Vector{Matrix{ComplexF64}} = map(method, Hpredict, H)
      m::Int64 , n::Int64 = size(basis)
      statistic::Matrix{Any} = Matrix{Any}(undef,m,n)
      I::Int64 = 1; v::Int64 = 1; w::Int64 = 1
      for i in 1:m
         J::Int64 = 1
         for j in 1:n
            v , w = size( first(ACE.evaluate(basis[i,j], first(configs))).val )
            Hblock = [ h[I:(I+v-1),J:(J+w-1)] for h in H]
            Hpblock = [ h[I:(I+v-1),J:(J+w-1)] for h in Hpredict]
            statistic[i,j] = method(Hpblock, Hblock)
            J += w
         end
         I += v
      end
      return statistic
   end
end




############


function depricated_test(coef::Matrix{Vector{ComplexF64}}, basis::Array{SymmetricBasis, 2}, 
   configs, H, method, eps::Float64=1e-15, intercept = false)
Hpredict::Vector{Matrix{ComplexF64}} = predict(coef, basis, configs, intercept)
E::Vector{Matrix{ComplexF64}} = [ zeros(size(H[i])) for i in 1:length(H)]
if method == "rmse"
E = ( H - Hpredict )
elseif method == "gabor"
E = [ ( Hpredict[i] ./ ( H[i] .+ eps ) ) .- 1 for i in 1:length(H) ]
end
m::Int64 , n::Int64 = size(basis)
statistic::Matrix{Float64} = zeros(m, n)
I::Int64 = 1; v::Int64 = 1; w::Int64 = 1
for i in 1:m
J::Int64 = 1
for j in 1:n
v , w = size( ACE.evaluate(basis[i,j], configs[1])[1].val )
e = [ err[I:(I+v-1),J:(J+w-1)] for err in E]
statistic[i,j] = norm(e)*( length(e)*v*w)^-0.5
J += w
end
I += v
end
return statistic
end




function parse_files_depricated(path, id, len, vol = 1, IJ = Nothing, rcut=5.0)

   # import file
   f = HDF5.h5open( path * "/" * string(id)  * ".h5", "r")
   matrices = [HDF5.read( f, string(i) ) for i in 0:(len-1)]
   HDF5.close(f)

   raw = JSON.parsefile(path * "/" * string(id) * ".json")
   m = length(raw); ns = [length(raw[string(i)]) for i in 0:(len-1) ]
   l = length(raw["1"][1])
   coords = [zeros(n,l) for n in ns]

   # convert coords to 
   for (ii,(key,value)) in enumerate(raw)
       i = parse(Int64,key)
       if i >= len # len er antallet matriser som skal hentes ut
           continue 
       end
       for (j, vec) in enumerate(value)
           coords[i+1][j,:] = Float64.( vec )
       end
   end


   # for hver index (0-99) må vi velge ut noe som er nærmere enn cutoff
   # remember to pick some coordinates that have nonzero cutoff.
   # this is a lazy method but avoids the boundaries of the material
   N = 4; 
   H::Vector{Matrix{ComplexF64}} = []
   idx = []
   R::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef,0)

   if IJ == Nothing
       IJ::Vector{Tuple{Int64,Int64}} = []
       for (i, coord) in enumerate(coords)
           I::Int64 = 0; J::Int64 = 0
           choice = [] # this will contain coordinates of Hamiltonian blocks
           index = []
           while length(choice) < vol # while I am yet to collect k suitable coordinates
               I, J = randperm(size(coord)[1])[1:2] # I try distinct coords (that produces offsite blocks)
               if LinearAlgebra.norm(coord[I,:] - coord[J,:]) < rcut # Only if atoms are close enough ...
                   append!(choice, [(I,J)] ) # ... their indices are added to the array of chosen coordinates
                   append!(index,[i])
               end
           end
           for (I,J) in choice
               append!(IJ,[(I,J)])
               append!(idx,index)
           end
       end
   end
   for (K,(I,J)) in enumerate(IJ)
       i = N*(I-1)+1
       j = N*(J-1)+1
       append!(H, [ matrices[K][i:(i+N-1), j:(j+N-1) ] ] ) # The hamiltonian-array is extended by our choices
       append!(R, [ coords[K]' ] ) # The hamiltonian-array is extended by our choices
   end


   
   Z = [zeros(Int64, 1) for i in 1:length(R)]

   for i in 1:length(R)
       n = length(R'[i][:,1]); 
       Z[i] = Int64.(14*ones(n))
   end
   

   # I assume that the cell does not change for each sample.
   jsoncell = isfile(path * "/" * string(id) * ".cell.json")
   stdcell = isfile(path * "/" * string(id) * ".cell")
   cell = [ zeros(3,3) for i in 1:length(H) ]
   
   if jsoncell & stdcell
       error("WHAT KIND OF CELL DO YOU WANT?")
   elseif stdcell 
       unitcell = eval(Meta.parse(read(path * "/" * string(id) * ".cell", String)))
       cell = [unitcell' for _ in 1:length(H)]
   elseif jsoncell
       raw = JSON.parsefile(path * "/" * string(id) * ".cell.json")
       for (ii, K) in enumerate(idx)
           value = raw[string(K-1)]
           for (j, vec) in enumerate(value)
               cell[ii][j,:] = Float64.( vec )'
           end
       end
   else 
       error("NO CELL FOUND!")
   end
   return H, R, IJ, cell, Z
end



end
