using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using Plots
pyplot()

using Zygote: dropgrad, ignore, Buffer

f_exact(x, y) = sinpi(x) * sinh(pi * y) / sinh(pi)
fx_exact(x, y) = cospi(x) * sinh(pi * y) * pi / sinh(pi)
fy_exact(x, y) = sinpi(x) * cosh(pi * y) * pi / sinh(pi)
fxy_exact(x, y) = cospi(x) * cosh(pi * y) * pi^2 / sinh(pi)
f_test(x::Vector) = [f_exact(x[1], x[2]),
                     fx_exact(x[1], x[2]),
                     fy_exact(x[1], x[2])]

const P1 = 0.21132486540518708  # Gauss Points on interval [0, 1]
const P2 = 0.7886751345948129
const PV1 = Float64[1, P1, P1^2, P1^3]
const PV2 = Float64[1, P2, P2^2, P2^3]
const DPV1 = Float64[0, 1, 2P1, 3P1^2]
const DPV2 = Float64[0, 1, 2P2, 3P2^2]
const DDPV1 = Float64[0, 0, 2, 6P1]
const DDPV2 = Float64[0, 0, 2, 6P2]

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

function e2n(ne, ng)
    quotient = div(ne - 1, ng)
    res = ne - ng * quotient
    n1 = quotient * (ng + 1) + res
    n2 = n1 + 1
    n4 = n2 + ng
    n3 = n4 + 1
    (n1, n2, n3, n4)
end

error(sol, nodes) = begin
    u = sol.minimizer
    v = hcat(([nodes[:, i] for i in 1:size(nodes, 2)] .|> f_test)...)
    sqrt.((mean(abs2, u - v, dims=2)))
end

error(result::Tuple) = begin
    error(result[1], result[3])
end

show_map(sol) = begin
    theme(:vibrant)
    u = sol.minimizer[1, :]
    ng = sqrt(length(u)) - 1 |> Integer
    umap = reshape(u, ng+1, ng+1)
    heatmap(umap, aspect_ratio=1, show=true, clim=(0, 1))
end

walls(NG) = begin
    upper_wall = ((NG+1)*NG+1):(NG+1)^2
    lower_wall = 1:(NG+1)
    left_wall = 1:(NG+1):(NG*(NG+1)+1)
    right_wall = (NG+1):(NG+1):(NG+1)^2
    (upper_wall, lower_wall, left_wall, right_wall)
end


function train(NG)
    side = NG^-1 * (0:NG)
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(3, (NG+1)^2)
    upper_wall, lower_wall, left_wall, right_wall = walls(NG)
    data[1, upper_wall] = nodes[1, upper_wall] .|> sinpi
    # data[3, upper_wall] = nodes[1, upper_wall] .|> (x -> pi * sinpi(x) / tanh(pi))

    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob = OptimizationProblem(opt_f, data, nodes)
    sol = solve(prob, LBFGS())
    (sol, prob, nodes)
end

function loss(data, nodes)
    ng = sqrt(size(data, 2)) - 1 |> Integer
    upper_wall, lower_wall, left_wall, right_wall = walls(ng)
    buf = Buffer(data)
    buf[:, :] = data[:, :]
    buf[1, left_wall] = dropgrad(data[1, left_wall])
    buf[1, right_wall] = dropgrad(data[1, right_wall])
    buf[1, lower_wall] = dropgrad(data[1, lower_wall])
    buf[1, upper_wall] = dropgrad(data[1, upper_wall])
    # buf[1, lower_wall] = dropgrad(data[1, lower_wall])
    data = copy(buf)

    sum([element_loss(i, ng, data, nodes) for i in 1:ng^2])
end

function element_loss(ne, ng, data, nodes)

    n1, n2, n3, n4 = e2n(ne, ng)
    Δx = nodes[1, n2] - nodes[1, n1]
    Δy = nodes[2, n4] - nodes[2, n1]
    ratio = Float64[1, Δx, Δy, Δx * Δy]
    f = @views [data[:, n1] data[:, n2] data[:, n3] data[:, n4]]
    g1 = 0.5Δx^-1 * (f[3, 2] - f[3, 1]) + 0.5Δy^-1 * (f[2, 4] - f[2, 1])
    g2 = 0.5Δx^-1 * (f[3, 2] - f[3, 1]) + 0.5Δy^-1 * (f[2, 3] - f[2, 2])
    g3 = 0.5Δx^-1 * (f[3, 3] - f[3, 4]) + 0.5Δy^-1 * (f[2, 3] - f[2, 2])
    g4 = 0.5Δx^-1 * (f[3, 3] - f[3, 4]) + 0.5Δy^-1 * (f[2, 4] - f[2, 1])
    ff = [f; [g1 g2 g3 g4]] |> vec
    α = reshape(Ainv * ff, 4, 4)
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
