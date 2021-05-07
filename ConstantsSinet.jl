module ConstantsSinet
using LinearAlgebra

export BD_SIZE, BATCH_SIZE, DIM, P_SIZE, PJ1, PJ2, PJ3, P_SIZE_FULL

const DIM = 2
const BATCH_SIZE = 100
const BD_SIZE = BATCH_SIZE |> sqrt |> ceil |> Int # boundary data batch size
const P_SIZE = 10
const P_SIZE_FULL = 2P_SIZE+4P_SIZE^2

const PJ1 = diagm(P_SIZE, P_SIZE_FULL, 0 => ones(Float32, P_SIZE))
const PJ2 = diagm(P_SIZE, P_SIZE_FULL, P_SIZE => ones(Float32, P_SIZE))
const PJ3 = diagm(4P_SIZE^2, P_SIZE_FULL, 2P_SIZE => ones(Float32, 4P_SIZE^2))

end
