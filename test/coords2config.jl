using goofy, LinearAlgebra
using ACE: CylindricalBondEnvelope
using Random

# denne testen lager relative koordinater på to måter: Først 
# ved å velge ut de relative koordinatene fra coords2configs 
# (test coordinates). Deretter ved å lage den manuelt (inkl 
# filtrere de som ligger utenfor sylinderen rundt RIJ) det er 
# ref coordinates

# a function that does orth projection of the vector u on the vector v
function proj(u, v)
    return ( dot(u,v)/norm(v)^2 )*v
end

# this function checks if R_IK is in the cylinder of norm R_IJ + 2 renv and radius renv
function in_cylinder(R_IK, R_IJ, rcut_length, renv_length )
    renv = R_IJ*renv_length/norm(R_IJ) # I give renv_length the same direction as R_IJ
    P = proj(R_IK, R_IJ)
    a = (  norm( R_IK - P) <= renv_length  ) # it is inside closed cylinder by radius
    b = (  dot(R_IJ, P + renv) >= 0 ) # the particle is outside the cylinder
    c = (  norm(P) <= norm(R_IJ + renv)  )
    return a & b & c
end

# this function is not needed now, but aligns a cylinder to the x-axis
function rotated_coords(R_IK, R_IJ)
    P = proj(R_IK,R_IJ)
    u = R_IJ/norm(R_IJ)
    w = (R_IK - P)/norm(R_IK - P)
    v = cross(w,u)/norm(cross(w,u))
    Ainv = zeros(3,3)
    Ainv[:,1] = u; Ainv[:,2] = v; Ainv[:,3] = w
    return inv(Ainv)
end


n = 200 # how many systems do you want to test for?
N = 5 # how many particles in a system?
rcut = 5.0 ; renv = rcut/2
envelope::CylindricalBondEnvelope = CylindricalBondEnvelope(rcut, renv, renv)
lat::Float64 = 5
cell = Matrix(lat*I, 3, 3)
a1,a2,a3 = [cell[:,x] for x in 1:size(cell,2)]
R = [lat*rand(N,3) for i in 1:n]
Z = Int64.(14*ones(N))
IJ::Vector{Tuple{Int64,Int64}} = [Tuple(randperm(5)[1:2]) for i in 1:n]


coords = [IJ, R]


# TEST
# I get relative coordinates using function under test that are of type :env
configs = coords2configs(coords, Z, envelope,  cell)
test_coords::Vector{Vector{Vector{Float64}}} = []
for config in configs
    test_coord = []
    for state in config.Xs
        if state.be == :env 
            append!(test_coord, [state.rr] ) 
        end
    end
    if ! isempty(test_coord)
        append!(test_coords, [test_coord])
    end

end


# REFERENCE
# Step 1: The appropriate R_IJ is determined
R_IJ = []
for (l, R_l) in enumerate(R)
    K = IJ[l][1]; L = IJ[l][2] ; R_I = R_l[K,:]
    candidates::Vector{Vector{Float64}} = []
    for i in [-2,-1,0,1,2]
        for j in [-2,-1,0,1,2]
            for k in [-2,-1,0,1,2]
                v = i*a1 + j*a2 + k*a3 + R[l][L,:] - R_I
                append!(candidates, [v])
            end
        end
    end
    idx = sortperm(norm.(candidates))
    append!(R_IJ, [candidates[idx][1]] )
end

# Step 2: I filter coordinates that (a) have rcut > 5.0 AND (b) are not in the cylinder AND 
# are distinct from R_IJ
ref_coords::Vector{Vector{Vector{Float64}}} = []
for l in 1:length(R)
    K = IJ[l][1]
    R_I = R[l][K,:]
    coords = R[l][1:end .!= K,:] # these are coordinates inside the unit cell, except R_I
    ref_coord::Vector{Vector{Float64}} = []
    for coord in [coords[x,:] for x in 1:size(coords,1)]
        eqiv::Vector{Vector{Float64}} = []
        for i in [-1,0,1]
            for j in [-1,0,1]
                for k in [-1,0,1]
                    R_IK = i*a1 + j*a2 + k*a3 + coord - R_I
                    if ! in_cylinder(R_IK, R_IJ[l], rcut, renv)
                        continue
                    end
                    if norm( R_IK - R_IJ[l] ) < 1e-15
                        continue
                    end
                    if ( norm(R_IK) > 5.0 )
                        continue
                    end
                    append!( eqiv, [ i*a1 + j*a2 + k*a3 + coord - R_I ] )
                end
            end
        end
        if isempty(eqiv)
            continue
        end
        append!(ref_coord, eqiv)
    end
    if ! isempty(ref_coord)
        append!(ref_coords, [ref_coord])
    end
end


e = sort.(test_coords) - sort.(ref_coords)
rmse = norm.(e)./length.(e)

# did the test pass?
println( "pass ", ! Bool( sum(rmse .> 1e-15 ) )  )

