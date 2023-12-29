
using goofy, JSON, HDF5, LinearAlgebra, Random
using ACE: BondEnvelope, CylindricalBondEnvelope


# basis config
# I set the degree, maxorder, cutoff and regularization parameter
rcut = 5.0; renv = rcut/2; 
file_ids = ["1","1"]
lambda = 1e-7
order = 1; degree = 5
len = 20
path = "/home/marius/Dokumenter/Skole/phd/goofy-e2e-data/e2e.h5"
# it is assumed that the order of the orbitals in the sub-blocks are the following:
# first come all the s-orbitals, then all the p-orbitals, then the d-orbitals and so on

_IJ = [ [(40, 35), (24, 43), (40, 37), (18, 44), (45, 23), (39, 57), (19, 69), (32, 28), 
        (58, 35), (32, 35), (6, 34), (53, 61), (47, 25), (30, 3), (61, 37), (44, 42),  
        (37, 16), (52, 61), (56, 37), (57, 50)], [(27, 56), (61, 41), (11, 15), (40, 44), 
        (36, 39), (20, 70), (28, 4), (30, 61), (56, 54), (69, 47), (62, 69), (19, 69), 
        (9, 15), (69, 70), (19, 20), (11, 15), (54, 53), (14, 17), (63, 56), (21, 69)] ]


_H = []; _R = []; _unitcell = []; _Z = []

for (i, file_id) in enumerate(file_ids)
    H, R, cell, Z = parse_files(path, _IJ[i], Vector(1:len) )
    append!(_R,[R]); append!(_H,[H]); append!(_unitcell,[cell]); append!(_Z,[Z])
end 

L_cfg = Dict(0=>1, 1=>1)  # r0cut rcut
ace_param = [degree, order, rcut, renv, L_cfg]
fit_param = [_H[1], lambda, lsqr_solver, false]
system = [_IJ[1], _R[1], _Z[1], _unitcell[1]]
c, _, __, basis, configs = train(system, ace_param, fit_param)



function rmse(X::Vector{Matrix{ComplexF64}}, Y::Vector{Matrix{ComplexF64}}) 
    Z = map( (x,y) -> x .- y, X, Y )
    N = length(Z)*length(first(Z))
    return ( real(dot( Z, Z ))/N )^0.5
end

function rel(X::Vector{Matrix{ComplexF64}}, Y::Vector{Matrix{ComplexF64}}) 
    Z = map( (x,y) -> x./ (y .+ 0) .- 1, X, Y )
    N = length(Z)*length(first(Z))
    return ( real(dot( Z, Z ))/N )^0.5
end



env = CylindricalBondEnvelope(rcut, renv, rcut/2)
test_configs = coords2configs([_IJ[2], _R[2]], _Z[2], env, _unitcell[2])
rmse_jig = test_setup(c, basis, rmse)
rel_jig = test_setup(c, basis, rel)


rmse_train = rmse_jig(_H[1], configs)
rmse_test = rmse_jig(_H[2], test_configs)

atol = 1e-16
rmse1 = (rmse_train - [ 0.0007602226785111889 0.00041656646394651266; 
                        0.00041656646394651266 0.0011093674568751575])
rmse2 = (rmse_test - [  0.02693919867893994 0.01437344862680931; 
                        0.01437344862680931 0.005193592832479996])
if norm(rmse1) > atol 
    @show rmse1 
end
if norm(rmse2) > atol 
    @show rmse2 
end

gabor_train = rel_jig(_H[1], configs)
gabor_test = rel_jig(_H[2], test_configs)



rmse3 = (gabor_train - [0.23278827039906394 0.45212357885746823; 
                        0.45212357885746823 92.6402035166913])
rmse4 = (gabor_test - [ 0.2773642842965894 0.7580229471196875; 
                        0.7580229471196875 8.28533180562432])
if norm(rmse3) > atol 
    @show rmse3 
end
if norm(rmse4) > atol 
    @show rmse4 
end

pass = (norm(rmse1) + norm(rmse2) + norm(rmse3) + norm(rmse4) < 4*atol)

@show pass