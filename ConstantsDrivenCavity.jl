module ConstantsDrivenCavity
using LinearAlgebra

export DIM, DIM_OUT, HIDDEN, BATCH_SIZE
export BD_SIZE, P_SIZE
export P1, P2, P3, P4
export PJ11, PJ12, PJ21, PJ22
export NU

### CONSTANTS
###
### Independent constants
const DIM = 2
const DIM_OUT = 1
const HIDDEN = 20
const BATCH_SIZE = 100
### Dependent constants
const BD_SIZE = BATCH_SIZE |> sqrt |> ceil |> Int # boundary data batch size
const P_SIZE = DIM * HIDDEN + HIDDEN + HIDDEN * 1 + 1
const P1 = DIM * HIDDEN
const P2 = HIDDEN
const P3 = HIDDEN * DIM_OUT
const P4 = DIM_OUT
const PJ11 = diagm(P1, P_SIZE, 0 => ones(Float32, P1))
const PJ12 = diagm(P2, P_SIZE, P1 => ones(Float32, P2))
const PJ21 = diagm(P3, P_SIZE, P1 + P2 => ones(Float32, P3))
const PJ22 = diagm(P4, P_SIZE, P1 + P2 + P3 => ones(Float32, P4))
const NU = 0.1f0
### END
###
end
