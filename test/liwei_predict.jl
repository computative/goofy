using goofy, JSON, HDF5, LinearAlgebra, Random

# I set the degree, maxorder, cutoff and regularization parameter
rcut = 5.0; renv = rcut/2; 
order = 1; degree = 3
lambda = 0*1e-12
len = 4


# H, R are collections of configurations, Z and cell are charge and unit cell for all configs

path = "/home/marius/Dokumenter/Skole/phd/goofy.git/test"
id = "1"
vol = 1

# import file
f = HDF5.h5open( path * "/" * id  * ".h5", "r")
matrices = [HDF5.read( f, string(i) ) for i in 0:(len-1)]
HDF5.close(f)

raw = JSON.parsefile(path * "/" * id * ".json")
m = length(raw); ns = [length(raw[string(i)]) for i in 0:(len-1) ]
l = length(raw["1"][1])
coords = [zeros(n,l) for n in ns]

# convert coords to 
for (ii,(key,value)) in enumerate(raw)
    i = parse(Int64,key)
    if i >= len
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
IJ = [(52, 16),(52, 16),(52, 16),(52, 16)]
for (K,(I,J)) in enumerate(IJ)
    i = N*(I-1)+1
    j = N*(J-1)+1
    append!(H, [ matrices[K][i:(i+N-1), j:(j+N-1) ] ] ) # The hamiltonian-array is extended by our choices
    append!(R, [ coords[K] ] ) # The hamiltonian-array is extended by our choices
end



Z = [zeros(Int64, 1) for i in 1:length(R)]

for i in 1:length(R)
    n = length(R[i][:,1]); 
    Z[i] = Int64.(14*ones(n))
end


# I assume that the cell does not change for each sample.
jsoncell = isfile(path * "/" * id * ".cell.json")
stdcell = isfile(path * "/" * id * ".cell")
cell = [ zeros(3,3) for i in 1:length(H) ]

if jsoncell & stdcell
    error("WHAT KIND OF CELL DO YOU WANT?")
elseif stdcell 
    unitcell = eval(Meta.parse(read(path * "/" * id * ".cell", String)))
    cell = [unitcell for _ in 1:length(H)]
elseif jsoncell
    raw = JSON.parsefile(path * "/" * id * ".cell.json")
    for (ii, K) in enumerate(idx)
        value = raw[string(K-1)]
        for (j, vec) in enumerate(value)
            cell[ii][j,:] = Float64.( vec )
        end
    end
else 
    error("NO CELL FOUND!")
end
L_cfg = Dict(0=>1, 1=>1)  # r0cut rcut
ace_param = [degree, order, rcut, renv, L_cfg]
fit_param = [H, lambda, inv_solver, false]
system = [IJ, R, Z, cell]




c, fitted, residuals, basis, configs = train(system, ace_param, fit_param)


# conclusions:
# The response vectors are the same. The design matrices are the same in the first 16 digits.
# The solvers are unstable. The determinant is 1e-120. Condition number is 

#println(real(c[1,1]))
#println(real(c[1,2]))
#println(real(c[2,1]))
#println(real(c[2,2]))

