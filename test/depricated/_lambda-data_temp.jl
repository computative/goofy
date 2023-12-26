using goofy
using JSON, HDF5, LinearAlgebra, Random, Plots


# I set the degree, maxorder, cutoff and regularization parameter
rcut = 5.0; renv = rcut/2; 
order = 1; degree = 15
stride = 1
lambda = 1e-20
k = 2^4

#p = plot()
#for k in ((2 .^ vec(0:5) )) 
#k = 2^4
#lambda = 1e-7
   #println("k = " , k)
   #lambdas = 10 .^ Vector(-30.0:3:20.0)
   #selector = 2
   #fig_data = []
   #for lambda in lambdas 
      println("lambda = " , lambda)

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

      N = 4; IJ::Vector{Tuple{Int64,Int64}} = []; 
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
      cell =eval(Meta.parse(read("/home/marius/Dokumenter/Skole/phd/goofy.git/test/1.cell",String)))

      # H, R are collections of configurations, Z and cell are charge and unit cell for all configs

      L_cfg = Dict(0=>1, 1=>1)

      #Hp = [ item' for item in H ]

      ace_param = [degree, order, rcut, renv, L_cfg]
      fit_param = [H[1:stride:end], lambda]
      system = [IJ[1:stride:end], R[1:stride:end], Z, cell]


      coef, fitted, e, basis, configs = train(system, ace_param, fit_param)

      rmse, Hpredict = test( coef, basis, configs, H[1:stride:end] )

      #println("rmse ", rmse[2])
      
      using Statistics
      println(std(coef[1,1])/mean(coef[1,1]))
      println(std(coef[1,2])/mean(coef[1,2]))
      #println(std(coef[2,1])/mean(coef[2,1]))
      #println(std(coef[2,2])/mean(coef[2,2]))




      4×4 Matrix{ComplexF64}:
  -1.88469+0.0im  -1.04039+0.0im   0.91008+0.0im  0.984451+0.0im
   1.04039+0.0im   0.46236+0.0im  -1.31893+0.0im  -1.42672+0.0im
  -0.91008+0.0im  -1.31893+0.0im  0.108307+0.0im   1.24802+0.0im
 -0.984451+0.0im  -1.42672+0.0im   1.24802+0.0im  0.304576+0.0im


 -1.88469-0.0im   1.04039-0.0im  -0.91008-0.0im  -0.984451-0.0im
 -1.04039-0.0im   0.46236-0.0im  -1.31893-0.0im   -1.42672-0.0im
  0.91008-0.0im  -1.31893-0.0im  0.108307-0.0im    1.24802-0.0im
 0.984451-0.0im  -1.42672-0.0im   1.24802-0.0im   0.304576-0.0im