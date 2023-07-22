using goofy, JSON, HDF5, LinearAlgebra, Random


# I set the degree, maxorder, cutoff and regularization parameter
rcut = 5.0; renv = rcut/2; 
order = 1; degree = 10
lambda = 1e-20

# it is assumed that the order of the orbitals in the sub-blocks are the following:
# first come all the s-orbitals, then all the p-orbitals, then the d-orbitals and so on

# basis config

f = HDF5.h5open("/home/marius/Dokumenter/Skole/phd/goofy.git/test/1.h5", "r")
matrices = [HDF5.read( f, string(i) ) for i in 0:99]
HDF5.close(f)

raw = JSON.parsefile("/home/marius/Dokumenter/Skole/phd/goofy.git/test/1.json")
m = length(raw); n = length(raw["1"]); l = length(raw["1"][1])
coords = [zeros(n,l) for i in 1:m]

for (key,value) in raw
   i = parse(Int64,key)
   for (j, vec) in enumerate(value)
      coords[i+1][j,:] = Float64.( vec )
   end
end


# for hver index (0-99) må vi velge ut noe som er nærmere enn cutoff
# remember to pick some coordinates that have nonzero cutoff.
# this is a lazy method but avoids the boundaries of the material

k = 10; N = 4; IJ::Vector{Tuple{Int64,Int64}} = []; 
H::Vector{Matrix{ComplexF64}} = []; R::Vector{Matrix{Float64}} = []
for (coord, matrix) in zip(coords, matrices)
   I::Int64 = 0; J::Int64 = 0
   choice = [] # this will contain coordinates of Hamiltonian blocks
   while length(choice) < k # while I am yet to collect k suitable coordinates
      I, J = randperm(size(coord)[1])[1:2] # I try distinct coords (that produces offsite blocks)
      if LinearAlgebra.norm(coord[I,:] - coord[J,:]) < rcut # Only if atoms are close enough ...
         append!(choice, [(I,J)] ) # ... their indices are added to the array of chosen coordinates
      end
   end
   for (I,J) in choice
      i = N*(I-1)+1
      j = N*(J-1)+1
      (N*(J-1)+1):( N*J )
      append!(H, [ matrix[i:(i+N-1), j:(j+N-1)] ] ) # The hamiltonian-array is extended by our choices
   end
   append!(IJ,choice)  # I extend the 'IJ'-array by the chosen coordinates
   append!(R, [coord for i in 1:k]) # R contains the coordinates of the systems
end




Z = Int64.(14*ones(length(IJ))) # I assume Z is the same for every system

# I assume that the cell does not change for each sample.
unitcell = eval(Meta.parse(read("/home/marius/Dokumenter/Skole/phd/goofy.git/test/1.cell", String)))

# H, R are collections of configurations, Z and cell are charge and unit cell for all configs

L_cfg = Dict(0=>1, 1=>1)
ace_param = [degree, order, rcut, renv, L_cfg]
fit_param = [H, lambda]
system = [IJ, R, Z, unitcell]


c, fitted, residuals, basis, configs = train(system, ace_param, fit_param)

rmse = test(c, basis, configs, H)





