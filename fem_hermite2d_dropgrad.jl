using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using Plots
pyplot()

using Zygote: dropgrad, ignore

const NG = 8                 # number of element on each side
const side = NG^-1 * (0:NG)
const nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)

const P1 = 0.21132486540518708  # Gauss Points on interval [0, 1]
const P2 = 0.7886751345948129
const PV1 = Float64[1, P1, P1^2, P1^3]
const PV2 = Float64[1, P2, P2^2, P2^3]
const DPV1 = Float64[0, 1, 2P1, 3P1^2]
const DPV2 = Float64[0, 1, 2P2, 3P2^2]
const DDPV1 = Float64[0, 0, 2, 6P1]
const DDPV2 = Float64[0, 0, 2, 6P2]

f_exact(x, y) = sinpi(x) * sinh(pi * (1-y)) / sinh(pi)
fx_exact(x, y) = cospi(x) * sinh(pi * (1-y)) * pi / sinh(pi)
fy_exact(x, y) = -sinpi(x) * cosh(pi * (1-y)) * pi / sinh(pi)
fxy_exact(x, y) = -cospi(x) * cosh(pi * (1-y)) * pi^2 / sinh(pi)
f_test(x::Vector) = [f_exact(x[1], x[2]), fx_exact(x[1], x[2]), fy_exact(x[1], x[2]), fxy_exact(x[1], x[2]),]

const Ainvtmp = Float64[
    1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 ;
    0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0 ;
    -3  3  0  0  -2  -1  0  0  0  0  0  0  0  0  0  0 ;
    2  -2  0  0  1  1  0  0  0  0  0  0  0  0  0  0 ;
    0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0 ;
    0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 ;
    0  0  0  0  0  0  0  0  -3  3  0  0  -2  -1  0  0 ;
    0  0  0  0  0  0  0  0  2  -2  0  0  1  1  0  0 ;
    -3  0  3  0  0  0  0  0  -2  0  -1  0  0  0  0  0 ;
    0  0  0  0  -3  0  3  0  0  0  0  0  -2  0  -1  0 ;
    9  -9  -9  9  6  3  -6  -3  6  -6  3  -3  4  2  2  1 ;
    -6  6  6  -6  -3  -3  3  3  -4  4  -2  2  -2  -2  -1  -1 ;
    2  0  -2  0  0  0  0  0  1  0  1  0  0  0  0  0 ;
    0  0  0  0  2  0  -2  0  0  0  0  0  1  0  1  0 ;
    -6  6  6  -6  -4  -2  4  2  -3  3  -3  3  -2  -1  -2  -1 ;
    4  -4  -4  4  2  2  -2  -2  2  -2  2  -2  1  1  1  1]

const Ainv = Ainvtmp[:, [4(0:3).+1; 4(0:3).+2; 4(0:3).+4; 4(0:3).+3]]

function e2n(ne)
    quotient = div(ne - 1, NG)
    res = ne - NG * quotient
    n1 = quotient * (NG + 1) + res
    n2 = n1 + 1
    n4 = n2 + NG
    n3 = n4 + 1
    (n1, n2, n3, n4)
end

function loss(data, p = nothing)
    #####
    ##### I need some kind of dropgrad code or ignore() code here!!!!
    #####
    sum([element_loss(i, data) for i in 1:NG^2])
end

function train()
    # data = zeros(4, (NG+1)^2)
    # data[1, 1:(NG+1)] = dropgrad(nodes[1, 1:(NG+1)] .|> sinpi)
    # data[3, 1:(NG+1)] = dropgrad(nodes[1, 1:(NG+1)] .|> (x -> -pi * sinpi(x) / tanh(pi)))
    # for i in 2:NG
    #     data[1, (i-1)*(NG+1)+1] = dropgrad(0.0)
    #     data[1, i*(NG+1)] = dropgrad(0.0)
    # end
    #
    data = zeros(4, (NG+1)^2)

    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob = OptimizationProblem(opt_f, data, nothing)
    sol = solve(prob, LBFGS())
    (sol, prob)
end

function element_loss(ne, data)
    n1, n2, n3, n4 = e2n(ne)
    Δx = nodes[1, n2] - nodes[1, n1]
    Δy = nodes[2, n4] - nodes[2, n1]
    ratio = Float64[1, Δx, Δy, Δx * Δy]
    f = @views [
        data[:, n1] .* ratio;
        data[:, n2] .* ratio;
        data[:, n3] .* ratio;
        data[:, n4] .* ratio
    ]
    α = reshape(Ainv * f, 4, 4)
    vv1 = dot(DDPV1', α, PV1)
    vv2 = dot(DDPV2', α, PV1)
    vv3 = dot(DDPV2', α, PV2)
    vv4 = dot(DDPV1', α, PV2)
    ww1 = dot(PV1', α, DDPV1)
    ww2 = dot(PV2', α, DDPV1)
    ww3 = dot(PV2', α, DDPV2)
    ww4 = dot(PV1', α, DDPV2)
    r1 = (vv1 + ww1)^2
    r2 = (vv2 + ww2)^2
    r3 = (vv3 + ww3)^2
    r4 = (vv4 + ww4)^2
    +(r1, r2, r3, r4) * Δx * Δy
end
