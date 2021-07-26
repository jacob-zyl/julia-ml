using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using Plots
pyplot()

using Zygote: dropgrad, ignore, Buffer, jacobian, hessian
using FastGaussQuadrature
const P, W = gausslegendre(2)

f_exact(x, y) = sinpi(x) * sinh(pi * y) / sinh(pi)
fx_exact(x, y) = cospi(x) * sinh(pi * y) * pi / sinh(pi)
fy_exact(x, y) = sinpi(x) * cosh(pi * y) * pi / sinh(pi)
fxy_exact(x, y) = cospi(x) * cosh(pi * y) * pi^2 / sinh(pi)
f_test(x::Vector) = [f_exact(x[1], x[2]), fx_exact(x[1], x[2]), fy_exact(x[1], x[2])]

e2n(ne, ng) = begin
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
    xs = ng^-1 * (0:ng)
    ys = ng^-1 * (0:ng)
    heatmap(
        xs, ys, umap', aspect_ratio=1, show=true #clim=(0, 1)
    )
end

walls(NG) = begin
    upper_wall = ((NG+1)*NG+1):(NG+1)^2
    lower_wall = 1:(NG+1)
    left_wall = 1:(NG+1):(NG*(NG+1)+1)
    right_wall = (NG+1):(NG+1):(NG+1)^2
    (upper_wall, lower_wall, left_wall, right_wall)
end

H1(x, y) = 0.0625 * (1 - x)^2 * (1 - y)^2 * (2 + x) * (2 + y)
H1(x) = H1(x[1], x[2])
H2(x, y) = 0.0625 * (1 - x)^2 * (1 - y)^2 * (x + 1) * (2 + y)
H2(x) = H2(x[1], x[2])
H3(x, y) = 0.0625 * (1 - x)^2 * (1 - y)^2 * (2 + x) * (y + 1)
H3(x) = H3(x[1], x[2])

H4(x, y) = 0.0625 * (1 + x)^2 * (1 - y)^2 * (2 - x) * (2 + y)
H4(x) = H4(x[1], x[2])
H5(x, y) = 0.0625 * (1 + x)^2 * (1 - y)^2 * (x - 1) * (2 + y)
H5(x) = H5(x[1], x[2])
H6(x, y) = 0.0625 * (1 + x)^2 * (1 - y)^2 * (2 - x) * (y + 1)
H6(x) = H6(x[1], x[2])

H7(x, y) = 0.0625 * (1 + x)^2 * (1 + y)^2 * (2 - x) * (2 - y)
H7(x) = H7(x[1], x[2])
H8(x, y) = 0.0625 * (1 + x)^2 * (1 + y)^2 * (x - 1) * (2 - y)
H8(x) = H8(x[1], x[2])
H9(x, y) = 0.0625 * (1 + x)^2 * (1 + y)^2 * (2 - x) * (y - 1)
H9(x) = H9(x[1], x[2])

H10(x, y) = 0.0625 * (1 - x)^2 * (1 + y)^2 * (2 + x) * (2 - y)
H10(x) = H10(x[1], x[2])
H11(x, y) = 0.0625 * (1 - x)^2 * (1 + y)^2 * (x + 1) * (2 - y)
H11(x) = H11(x[1], x[2])
H12(x, y) = 0.0625 * (1 - x)^2 * (1 + y)^2 * (2 + x) * (y - 1)
H12(x) = H12(x[1], x[2])

H(x) = [H1(x), H2(x), H3(x), H4(x), H5(x), H6(x), H7(x), H8(x), H9(x), H10(x), H11(x), H12(x)]

P1 = [P[1], P[1]]
P2 = [P[2], P[1]]
P3 = [P[2], P[2]]
P4 = [P[1], P[2]]

const Hi1 = H(P1)
const Hi2 = H(P2)
const Hi3 = H(P3)
const Hi4 = H(P4)

tmp = jacobian(H, P1)[1]
const Hxi1 = tmp[:, 1]
const Hyi1 = tmp[:, 2]

tmp = jacobian(H, P2)[1]
const Hxi2 = tmp[:, 1]
const Hyi2 = tmp[:, 2]

tmp = jacobian(H, P3)[1]
const Hxi3 = tmp[:, 1]
const Hyi3 = tmp[:, 2]

tmp = jacobian(H, P4)[1]
const Hxi4 = tmp[:, 1]
const Hyi4 = tmp[:, 2]

const Hxxi1 = map(f -> vec(hessian(f, P1))[1], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])
const Hyyi1 = map(f -> vec(hessian(f, P1))[4], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])
const Hxyi1 = map(f -> vec(hessian(f, P1))[2], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])

const Hxxi2 = map(f -> vec(hessian(f, P2))[1], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])
const Hyyi2 = map(f -> vec(hessian(f, P2))[4], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])
const Hxyi2 = map(f -> vec(hessian(f, P2))[2], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])

const Hxxi3 = map(f -> vec(hessian(f, P3))[1], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])
const Hyyi3 = map(f -> vec(hessian(f, P3))[4], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])
const Hxyi3 = map(f -> vec(hessian(f, P3))[2], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])

const Hxxi4 = map(f -> vec(hessian(f, P4))[1], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])
const Hyyi4 = map(f -> vec(hessian(f, P4))[4], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])
const Hxyi4 = map(f -> vec(hessian(f, P4))[2], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12])

function train(NG)
    side = NG^-1 * (0:NG)
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(3, (NG+1)^2)
    upper_wall, lower_wall, left_wall, right_wall = walls(NG)
    # data[1, upper_wall] = nodes[1, upper_wall] .|> sinpi
    data[3, upper_wall] = nodes[1, upper_wall] .|> (x -> pi * sinpi(x) / tanh(pi))
    # data[1, lower_wall] = nodes[1, lower_wall] .|> sinpi
    # data[1, left_wall] = -nodes[2, left_wall] .|> sinpi
    # data[1, right_wall] = 1 .- nodes[2, right_wall] .|> sinpi

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
    buf[1, lower_wall] = dropgrad(data[1, lower_wall])
    # buf[1, upper_wall] = dropgrad(data[1, upper_wall])
    buf[1, left_wall] = dropgrad(data[1, left_wall])
    buf[1, right_wall] = dropgrad(data[1, right_wall])
    buf[3, upper_wall] = dropgrad(data[3, upper_wall])
    data = copy(buf)

    sum([element_loss(i, ng, data, nodes) for i in 1:ng^2])
end

function element_loss(ne, ng, data, nodes)

    n1, n2, n3, n4 = e2n(ne, ng)

    Δx = nodes[1, n2] - nodes[1, n1]
    Δy = nodes[2, n4] - nodes[2, n1]

    ratio = Float64[1, 0.5Δx, 0.5Δy]
    f = @views [
        data[:, n1] .* ratio;
        data[:, n2] .* ratio;
        data[:, n3] .* ratio;
        data[:, n4] .* ratio
    ]
    f1 = dot(f, Hi1)
    f2 = dot(f, Hi2)
    f3 = dot(f, Hi3)
    f4 = dot(f, Hi4)
    fx1 = dot(f, Hxi1) * 2 * Δx^-1
    fx2 = dot(f, Hxi2) * 2 * Δx^-1
    fx3 = dot(f, Hxi3) * 2 * Δx^-1
    fx4 = dot(f, Hxi4) * 2 * Δx^-1
    fy1 = dot(f, Hyi1) * 2 * Δy^-1
    fy2 = dot(f, Hyi2) * 2 * Δy^-1
    fy3 = dot(f, Hyi3) * 2 * Δy^-1
    fy4 = dot(f, Hyi4) * 2 * Δy^-1
    fxx1 = dot(f, Hxxi1) * 4 * Δx^-2
    fxx2 = dot(f, Hxxi2) * 4 * Δx^-2
    fxx3 = dot(f, Hxxi3) * 4 * Δx^-2
    fxx4 = dot(f, Hxxi4) * 4 * Δx^-2
    fyy1 = dot(f, Hyyi1) * 4 * Δy^-2
    fyy2 = dot(f, Hyyi2) * 4 * Δy^-2
    fyy3 = dot(f, Hyyi3) * 4 * Δy^-2
    fyy4 = dot(f, Hyyi4) * 4 * Δy^-2
    fxy1 = dot(f, Hxyi1) * 4 * Δx^-1 * Δy^-1
    fxy2 = dot(f, Hxyi2) * 4 * Δx^-1 * Δy^-1
    fxy3 = dot(f, Hxyi3) * 4 * Δx^-1 * Δy^-1
    fxy4 = dot(f, Hxyi4) * 4 * Δx^-1 * Δy^-1
    r1 = (fxx1 + fyy1)^2
    r2 = (fxx2 + fyy2)^2
    r3 = (fxx3 + fyy3)^2
    r4 = (fxx4 + fyy4)^2
    +(r1, r2, r3, r4) * Δx * Δy * 0.25
end
