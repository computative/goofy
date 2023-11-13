using ACE, StaticArrays
using ACE: Utils.RnYlm_1pbasis, SphericalMatrix, SymmetricBasis, State, ACEConfig
using ACE: BondEnvelope, CylindricalBondEnvelope

# Offsite Basis construction
## TODO: if you need onsite, you will need to construct a onsite basis here
r0 = 2.5
rin = 0.5 * r0
rcut = 10.0
renv = 10-rcut/2
env = ACE.CylindricalBondEnvelope(rcut,renv,renv)

ord = 3
maxdeg = 20
Bsel = SimpleSparseBasis(ord, maxdeg)

# oneparticle basis
RnYlm = RnYlm_1pbasis(; maxdeg = maxdeg, r0 = r0, trans = PolyTransform(1, 1/r0^2), rcut = sqrt(renv^2+(rcut+renv)^2), rin = rin, pcut = 2)
#ACE.init1pspec!(RnYlm, Bsel)
B1p_env = env * RnYlm
#ACE.init1pspec!(B1p_env, Bsel)
B1p = ACE.Categorical1pBasis([:bond, :env]; varsym = :be, idxsym=:be) * B1p_env
#ACE.init1pspec!(B1p, Bsel)
# SymmetricBasis
L1 = 0
L2 = 0
φ = SphericalMatrix(L1, L2; T = ComplexF64)
basis = SymmetricBasis(φ, B1p, Bsel)

# Configuration
## For onsite, all state in a configuration look like this R_IK = ACE.State(rr = SVector{3,Float64}(rand(3))), rr = r_K- r_I;

rr0 = SVector{3,Float64}(rand(3))
R_IJ = ACE.State(rr = SVector{3,Float64}(rand(3)), rr0 = rr0, be = :bond) 
R_IK = ACE.State(rr = SVector{3,Float64}(rand(3)), rr0 = rr0, be = :env) 
Config = ACEConfig([R_IJ,R_IK])

# Evaluation
BB = ACE.evaluate(basis,Config)
size(BB)
# Model
c = rand(length(basis))
SModel = ACE.LinearACEModel(basis,c)
ACE.evaluate(SModel, Config)

sum([BB[i] * c[i] for i = 1:length(c)]) == ACE.evaluate(SModel, Config)
