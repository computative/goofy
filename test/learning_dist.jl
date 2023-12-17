using goofy, JSON, HDF5, LinearAlgebra, ACE, Random
using ACE: BondEnvelope, CylindricalBondEnvelope, evaluate

# learning curves from the terminal
# execute with julia --project=. test/learning.jl n m l u v w k  (where n and m are the id of the training and testing data. 
# l is regularization parameter, u and v are the order and degree respectively
# While k is the number of observations that should be drawn from each full size hamiltonian matrix and w is the number of matrces to sample


# basis config
# I set the degree, maxorder, cutoff and regularization parameter
rcut = 5.0; renv = rcut/2; 
path = "/home/marius/Dokumenter/Skole/phd/goofy.git/test"

file_ids = ARGS[1:2]
lambda = parse(Float64, ARGS[3])
order = parse(Int64, ARGS[4]); degree = parse(Int64, ARGS[5])
lens = parse.(Int, split(chop(ARGS[6]; head=1, tail=1), ','))
vol = parse(Int64, ARGS[7])


# it is assumed that the order of the orbitals in the sub-blocks are the following:
# first come all the s-orbitals, then all the p-orbitals, then the d-orbitals and so on




for len in lens
    println("len: ", len)
    _H = []; _R = []; _unitcell = []; _Z = []; _IJ = []
    for (i, file_id) in enumerate(file_ids)
        H, R, IJ, cell, Z = parse_files(path, file_id, len)
        append!(_R,[R]); append!(_H,[H]); append!(_unitcell,[cell]); append!(_Z,[Z]); append!(_IJ,[IJ]); 
    end

    L_cfg = Dict(0=>1, 1=>1)  # r0cut rcut
    ace_param = [degree, order, rcut, renv, L_cfg]
    fit_param = [_H[1], lambda, "none"]
    system = [_IJ[1], _R[1], _Z[1], _unitcell[1]]
    c, fitted, residuals, basis, configs = train(system, ace_param, fit_param)

    for M in 1:2 # M mode 
        eps = 0.0
        if M == 2
            mode = "testing"
        elseif M == 1
            mode = "training"
        else
            error("UNKNOWN MODE")
        end
        cfgs = coords2configs([_IJ[M], _R[M]], _Z[M], CylindricalBondEnvelope(rcut, renv, rcut/2), _unitcell[M])
        Hpredict = predict(c, basis, cfgs)

        # these plots are as a function of separation
        x1 = [ norm(_R[M][i][_IJ[M][i][2],:] - _R[M][i][_IJ[M][i][1],:]) for i in 1:size(_R[M])[1] ]
        _y1 = [ abs.( Hpredict[i] ./ ( _H[M][i] .+ eps )  .- 1) for i in 1:length(_H[M]) ]
        m::Int64 , n::Int64 = size(basis)
        
        y1 = zeros(len, m, n)
        for k in 1:len
            I = 1; v = 1; w  = 1
            for i in 1:m
                J = 1
                for j in 1:n
                    v , w = size( ACE.evaluate(basis[i,j], configs[1])[1].val )
                    e = _y1[k][I:(I+v-1),J:(J+w-1)]
                    y1[k,i,j] = norm(e)*( v*w)^-0.5
                    J += w
                end
                I += v
            end
        end
        


        # these plots are as a function of matrix element size
        _x2 = [ abs.( _H[M][i] ) for i in 1:size(_R[M])[1] ]
        _y2 = [ abs.( Hpredict[i] ./ ( _H[M][i] .+ eps )  .- 1 ) for i in 1:length(_H[M]) ]
        x2 = [ _x2[i][j,k] for i in 1:length(_x2), j in 1:2, k in 1:2 ]


        y2 = zeros(len, m, n)
        for k in 1:len
            I = 1; v = 1; w = 1
            for i in 1:m
                J = 1
                for j in 1:n
                    v , w = size( ACE.evaluate(basis[i,j], configs[1])[1].val )
                    e = _y2[k][I:(I+v-1),J:(J+w-1)]
                    y2[k,i,j] = norm(e)*( v*w )^-0.5
                    J += w
                end
            I += v
            end
        end

        #y2 = [ statistic[i][j,k] for i in 1:length(statistic), j in 1:2, k in 1:2 ]
        println( mode, " i ", 1, " j ", 1, " x1 ", x1 )
        for i in 1:2
            for j in 1:2
                println(mode, " i ", i, " j ", j, " y1 ", y1[:,i,j])
            end
        end
        for i in 1:2
            for j in 1:2
                println(mode, " i ", i, " j ", j, " x2 ", x2[:,i,j])
                println(mode, " i ", i, " j ", j, " y2 ", y2[:,i,j])
            end
        end
    end
end


