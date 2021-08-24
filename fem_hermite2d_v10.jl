using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using Plots
pyplot()

using Zygote: dropgrad, ignore, Buffer, jacobian, hessian

using JLD

train() = begin
    mesh = load("mesh.jld")
    data = load("data.jld", "data")
    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())

    dt = 0.01
    for iteration in 1:10
        prob = OptimizationProblem(opt_f, data, (dt, mesh, data))
        sol = solve(prob, ConjugateGradient())
        data = sol.minimizer
        @save "data"*(@sprintf "%04i" iteration)*".jld" data
    end
end

loss(data, fem_dict) = begin
    dt, mesh, data_init = fem_dict
    ng = mesh["ng"]
    ne = mesh["ne"]
    nodes = mesh["nodes"]
    elnodes = mesh["elnodes"]
    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    buf = Buffer(data)
    buf[:, :] = data[:, :]
    buf[1, lower_wall] = dropgrad(data[1, lower_wall])
    buf[1, upper_wall] = dropgrad(data[1, upper_wall])
    buf[1, left_wall] = dropgrad(data[1, left_wall])
    buf[1, right_wall] = dropgrad(data[1, right_wall])
    data = copy(buf)

    sum = 0
    for iters in 1:ne
        indice = elnodes[:, iters]
        elnode = @views nodes[:, indice]
        eldata = @views data[:, indice]
        elinit = @views data_init[:, indice]
        sum += element_loss2(elnode, eldata, elinit)
    end
    sum
end

element_loss2(nodes, data, init) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]
    f = data .* ratio
    finit = init .* ratio

    f1 = dot(f, Hi1)
    f2 = dot(f, Hi2)
    f3 = dot(f, Hi3)
    f4 = dot(f, Hi4)
    finit1 = dot(finit, Hi1)
    finit2 = dot(finit, Hi2)
    finit3 = dot(finit, Hi3)
    finit4 = dot(finit, Hi4)
    fxx1 = dot(f, Hxxi1) * 4 * Δx^-2
    fxx2 = dot(f, Hxxi2) * 4 * Δx^-2
    fxx3 = dot(f, Hxxi3) * 4 * Δx^-2
    fxx4 = dot(f, Hxxi4) * 4 * Δx^-2
    fyy1 = dot(f, Hyyi1) * 4 * Δy^-2
    fyy2 = dot(f, Hyyi2) * 4 * Δy^-2
    fyy3 = dot(f, Hyyi3) * 4 * Δy^-2
    fyy4 = dot(f, Hyyi4) * 4 * Δy^-2
    finitxx1 = dot(finit, Hxxi1) * 4 * Δx^-2
    finitxx2 = dot(finit, Hxxi2) * 4 * Δx^-2
    finitxx3 = dot(finit, Hxxi3) * 4 * Δx^-2
    finitxx4 = dot(finit, Hxxi4) * 4 * Δx^-2
    finityy1 = dot(finit, Hyyi1) * 4 * Δy^-2
    finityy2 = dot(finit, Hyyi2) * 4 * Δy^-2
    finityy3 = dot(finit, Hyyi3) * 4 * Δy^-2
    finityy4 = dot(finit, Hyyi4) * 4 * Δy^-2
    α = 1.0
    r1 = (α * (fxx1 + fyy1) + (1.0 - α) * (finitxx1 + finityy1) + 10.0 * (finit1 - f1))^2
    r2 = (α * (fxx2 + fyy2) + (1.0 - α) * (finitxx2 + finityy2) + 10.0 * (finit2 - f2))^2
    r3 = (α * (fxx3 + fyy3) + (1.0 - α) * (finitxx3 + finityy3) + 10.0 * (finit3 - f3))^2
    r4 = (α * (fxx4 + fyy4) + (1.0 - α) * (finitxx4 + finityy4) + 10.0 * (finit4 - f4))^2
    +(r1, r2, r3, r4) * Δx * Δy * 0.25
end

f_exact(x, y) = sinpi(x) * sinh(pi * y) / sinh(pi)
fx_exact(x, y) = cospi(x) * sinh(pi * y) * pi / sinh(pi)
fy_exact(x, y) = sinpi(x) * cosh(pi * y) * pi / sinh(pi)
fxy_exact(x, y) = cospi(x) * cosh(pi * y) * pi^2 / sinh(pi)
f_test(x::Vector) = [f_exact(x[1], x[2]), fx_exact(x[1], x[2]), fy_exact(x[1], x[2]), fxy_exact(x[1], x[2])]

error(sol, mesh) = begin
    nodes = mesh["nodes"]
    u = sol.minimizer
    v = hcat(([nodes[:, i] for i in 1:size(nodes, 2)] .|> f_test)...)
    sqrt.((mean(abs2, u - v, dims=2)))
end

error(result::Tuple) = begin
    error(result[1], result[3])
end


show_map(sol) = begin
    u = sol.minimizer[1, :]
    show_map(u)
end

show_map(u::Array) = begin
    theme(:vibrant)
    ng = sqrt(length(u)) - 1 |> Integer
    umap = reshape(u, ng+1, ng+1)
    xs = ng^-1 * (0:ng)
    ys = ng^-1 * (0:ng)
    heatmap(
        xs, ys, umap', aspect_ratio=1, show=true, #clim=(0, 1)
    )
end


show_map(sol, nodes) = begin
    theme(:vibrant)
    u = sol.minimizer[1, :]
    v = hcat(([nodes[:, i] for i in 1:size(nodes, 2)] .|> f_test)...)
    ng = sqrt(length(u)) - 1 |> Integer
    error_map = map(1:ng^2) do x
        element_loss(x, ng, sol.minimizer, nodes) |> sqrt
    end
    p1 = show_map(sol)
    p2 = heatmap(log10.(reshape(error_map, ng, ng))', aspect_ratio=1)
    plot(p1, p2)
end

walls(NG) = begin
    upper_wall = ((NG+1)*NG+1):(NG+1)^2
    lower_wall = 1:(NG+1)
    left_wall = 1:(NG+1):(NG*(NG+1)+1)
    right_wall = (NG+1):(NG+1):(NG+1)^2
    (upper_wall, lower_wall, left_wall, right_wall)
end







using FastGaussQuadrature
const P, W = gausslegendre(2)

# basis related to point (-1, -1)
H1(x, y) = 0.0625 * (1 - x)^2 * (1 - y)^2 * (2 + x) * (2 + y)
H2(x, y) = 0.0625 * (1 - x)^2 * (1 - y)^2 * (x + 1) * (2 + y)
H3(x, y) = 0.0625 * (1 - x)^2 * (1 - y)^2 * (2 + x) * (y + 1)
H4(x, y) = 0.0625 * (1 - x)^2 * (1 - y)^2 * (x + 1) * (y + 1)

# basis related to point (1, -1)
H5(x, y) = 0.0625 * (1 + x)^2 * (1 - y)^2 * (2 - x) * (2 + y)
H6(x, y) = 0.0625 * (1 + x)^2 * (1 - y)^2 * (x - 1) * (2 + y)
H7(x, y) = 0.0625 * (1 + x)^2 * (1 - y)^2 * (2 - x) * (y + 1)
H8(x, y) = 0.0625 * (1 + x)^2 * (1 - y)^2 * (x - 1) * (y + 1)

# basis related to point (1, 1)
H9(x, y) = 0.0625 * (1 + x)^2 * (1 + y)^2 * (2 - x) * (2 - y)
H10(x, y) = 0.0625 * (1 + x)^2 * (1 + y)^2 * (x - 1) * (2 - y)
H11(x, y) = 0.0625 * (1 + x)^2 * (1 + y)^2 * (2 - x) * (y - 1)
H12(x, y) = 0.0625 * (1 + x)^2 * (1 + y)^2 * (x - 1) * (y - 1)

# basis related to point (-1, 1)
H13(x, y) = 0.0625 * (1 - x)^2 * (1 + y)^2 * (2 + x) * (2 - y)
H14(x, y) = 0.0625 * (1 - x)^2 * (1 + y)^2 * (x + 1) * (2 - y)
H15(x, y) = 0.0625 * (1 - x)^2 * (1 + y)^2 * (2 + x) * (y - 1)
H16(x, y) = 0.0625 * (1 - x)^2 * (1 + y)^2 * (x + 1) * (y - 1)

H1(x) = H1(x[1], x[2])
H2(x) = H2(x[1], x[2])
H3(x) = H3(x[1], x[2])
H4(x) = H4(x[1], x[2])
H5(x) = H5(x[1], x[2])
H6(x) = H6(x[1], x[2])
H7(x) = H7(x[1], x[2])
H8(x) = H8(x[1], x[2])
H9(x) = H9(x[1], x[2])
H10(x) = H10(x[1], x[2])
H11(x) = H11(x[1], x[2])
H12(x) = H12(x[1], x[2])
H13(x) = H13(x[1], x[2])
H14(x) = H14(x[1], x[2])
H15(x) = H15(x[1], x[2])
H16(x) = H16(x[1], x[2])

H(x) = [H1(x), H2(x), H3(x), H4(x), H5(x), H6(x), H7(x), H8(x), H9(x), H10(x), H11(x), H12(x), H13(x), H14(x), H15(x), H16(x)]

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

const Hxxi1 = map(f -> vec(hessian(f, P1))[1], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])
const Hyyi1 = map(f -> vec(hessian(f, P1))[4], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])
const Hxyi1 = map(f -> vec(hessian(f, P1))[2], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])

const Hxxi2 = map(f -> vec(hessian(f, P2))[1], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])
const Hyyi2 = map(f -> vec(hessian(f, P2))[4], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])
const Hxyi2 = map(f -> vec(hessian(f, P2))[2], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])

const Hxxi3 = map(f -> vec(hessian(f, P3))[1], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])
const Hyyi3 = map(f -> vec(hessian(f, P3))[4], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])
const Hxyi3 = map(f -> vec(hessian(f, P3))[2], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])

const Hxxi4 = map(f -> vec(hessian(f, P4))[1], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])
const Hyyi4 = map(f -> vec(hessian(f, P4))[4], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])
const Hxyi4 = map(f -> vec(hessian(f, P4))[2], [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16])

gen() = begin
    # fem_dict = load("prob.jld")
    ng = 10
    nn = (ng + 1)^2
    ne = ng^2

    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    side = ng^-1 * (0:ng) |> collect
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(4, nn)

    data[1, upper_wall] = nodes[1, upper_wall] .|> sinpi
    # data[3, upper_wall] = (nodes[1, upper_wall] .|> sinpi) .* (pi / tanh(pi))
    # data[1, lower_wall] = nodes[1, lower_wall] .|> sinpi
    # data[1, left_wall] = -nodes[2, left_wall] .|> sinpi
    # data[1, right_wall] = 1 .- nodes[2, right_wall] .|> sinpi

    elnodes = hcat(map(p -> e2nvec(p, ng), 1:ne)...)

    @save "mesh.jld" ng ne nn elnodes nodes
    @save "data.jld" data
end

e2n(ne, ng) = begin
    quotient = div(ne - 1, ng)
    res = ne - ng * quotient
    n1 = quotient * (ng + 1) + res
    n2 = n1 + 1
    n4 = n2 + ng
    n3 = n4 + 1
    (n1, n2, n3, n4)
end
e2nvec(ne, ng) = begin
    quotient = div(ne - 1, ng)
    res = ne - ng * quotient
    n1 = quotient * (ng + 1) + res
    n2 = n1 + 1
    n4 = n2 + ng
    n3 = n4 + 1
    [n1, n2, n3, n4]
end
