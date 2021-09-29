using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using CairoMakie
using Zygote: dropgrad, ignore, Buffer, jacobian, hessian
using ForwardDiff: derivative
using JLD
using FastGaussQuadrature
using Quadrature, HCubature

const NK = 4

gen(ng = 7) = begin
    # fem_dict = load("prob.jld")
    nn = (ng + 1)^2
    ne = ng^2

    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    side = ng^-1 * (0:ng) |> collect
    # side = @. 0.5 - 0.5cos(pi*side)
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(4, nn)
    data[1, :] = ones(nn)

    data[1, upper_wall] = nodes[1, upper_wall] .|> sinpi
    # data[3, upper_wall] = (nodes[1, upper_wall] .|> sinpi) .* (pi / tanh(pi))
    data[1, lower_wall] = nodes[1, lower_wall] .|> zero
    data[1, left_wall] = -nodes[2, left_wall] .|> zero
    data[1, right_wall] = 1 .- nodes[2, right_wall] .|> zero

    elnodes = hcat(map(p -> e2nvec(p, ng), 1:ne)...)

    @save "heat2d-steady/mesh.jld" ng ne nn elnodes nodes
    @save "heat2d-steady/data.jld" data
end

train(ng = 7) = begin
    gen(ng)
    mesh = load("heat2d-steady/mesh.jld")
    data = load("heat2d-steady/data.jld", "data")
    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())

    prob = OptimizationProblem(opt_f, data, mesh)
    # sol = solve(prob, ConjugateGradient())
    sol = solve(prob, BFGS())
    data = sol.minimizer
    @save "heat2d-steady/result_"*(@sprintf "%03i" ng)*".jld" data mesh
end

loss(data, mesh) = begin
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
        sum += element_loss(elnode, eldata)
    end
    sum
end

element_loss(nodes, data) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]
    f = data .* ratio

    u = Hi * vec(f)

    # coefficients below are from element governing equation
    uxx = Hxxi * vec(f) * 4 * Δx^-2
    uyy = Hyyi * vec(f) * 4 * Δy^-2
    r = @. (uxx + uyy)^2
    weights ⋅ r
end

element_loss_conservative(nodes, data) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]
    f = data .* ratio

    # damn... magic numbers
    Nindex = [15, 16, 11, 12]
    Sindex = [3, 4, 7, 8]
    Eindex = [6, 8, 10, 12]
    Windex = [2, 4, 14, 16]
    fN = @view f[Nindex]
    fS = @view f[Sindex]
    fW = @view f[Windex]
    fE = @view f[Eindex]
    res = Wthi ⋅ (fN + fE - fS - fW)
    res^2
end

f_exact(x, y) = sinpi(x) * sinh(pi * y) / sinh(pi)
fx_exact(x, y) = cospi(x) * sinh(pi * y) * pi / sinh(pi)
fy_exact(x, y) = sinpi(x) * cosh(pi * y) * pi / sinh(pi)
fxy_exact(x, y) = cospi(x) * cosh(pi * y) * pi^2 / sinh(pi)
f_test(x) = [f_exact(x[1], x[2]), fx_exact(x[1], x[2]), fy_exact(x[1], x[2]), fxy_exact(x[1], x[2])]

error(result) = error(load(result, "data"), load(result, "mesh"))
error(data, mesh) = begin
    integrand(x, p) = (interpolate(data, mesh)(x[1], x[2]) - f_exact(x[1], x[2]))^2
    prob = QuadratureProblem(integrand, zeros(2), ones(2))
    sol = solve(prob, HCubatureJL())
    sol.u |> sqrt
end

show_map(result) = show_map(load(result, "data"), load(result, "mesh"))
show_map(data, mesh) = begin
    ng = sqrt(length(size(data, 2))) - 1 |> Integer
    xs = 0:0.005:1
    ys = 0:0.005:1
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    heatmap!(ax, xs, ys, interpolate(data, mesh), colormap=:bwr)
    fig
end

show_loss_map(data, mesh) = begin
    ng = mesh["ng"]
    ne = mesh["ne"]
    nodes = mesh["nodes"]
    elnodes = mesh["elnodes"]
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    for iters in 1:ne
        indice = elnodes[:, iters]
        elnode = @views nodes[:, indice]
        eldata = @views data[:, indice]
        append!(xs, mean(elnode, dims=2)[1])
        append!(ys, mean(elnode, dims=2)[2])
        append!(zs, element_loss(elnode, eldata))
    end
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    heatmap(xs, ys, zs)
    #Colorbar(fig[1, 2], limits = (minimum(zs), maximum(zs)))
    fig
end


walls(NG) = begin
    upper_wall = ((NG+1)*NG+1):(NG+1)^2
    lower_wall = 1:(NG+1)
    left_wall = 1:(NG+1):(NG*(NG+1)+1)
    right_wall = (NG+1):(NG+1):(NG+1)^2
    (upper_wall, lower_wall, left_wall, right_wall)
end

# correctness of code blow verified.

const P, W = gausslegendre(NK)

const points = tuple.(P', P)
const weights = kron(W, W)

H1(x) = (1.0 - x)^2 * (2.0 + x) * 0.25
H2(x) = (1.0 - x)^2 * (x + 1.0) * 0.25
H3(x) = (1.0 + x)^2 * (2.0 - x) * 0.25
H4(x) = (1.0 + x)^2 * (x - 1.0) * 0.25

Hx1(x) = derivative(p -> H1(p), x)
Hx2(x) = derivative(p -> H2(p), x)
Hx3(x) = derivative(p -> H3(p), x)
Hx4(x) = derivative(p -> H4(p), x)

Hxx1(x) = derivative(p -> Hx1(p), x)
Hxx2(x) = derivative(p -> Hx2(p), x)
Hxx3(x) = derivative(p -> Hx3(p), x)
Hxx4(x) = derivative(p -> Hx4(p), x)

H(p) = [H1(p[1])*H1(p[2]), H2(p[1])*H1(p[2]), H1(p[1])*H2(p[2]), H2(p[1])*H2(p[2]),
        H3(p[1])*H1(p[2]), H4(p[1])*H1(p[2]), H3(p[1])*H2(p[2]), H4(p[1])*H2(p[2]),
        H3(p[1])*H3(p[2]), H4(p[1])*H3(p[2]), H3(p[1])*H4(p[2]), H4(p[1])*H4(p[2]),
        H1(p[1])*H3(p[2]), H2(p[1])*H3(p[2]), H1(p[1])*H4(p[2]), H2(p[1])*H4(p[2])]'

Hxx(p) = [Hxx1(p[1])*H1(p[2]), Hxx2(p[1])*H1(p[2]), Hxx1(p[1])*H2(p[2]), Hxx2(p[1])*H2(p[2]),
          Hxx3(p[1])*H1(p[2]), Hxx4(p[1])*H1(p[2]), Hxx3(p[1])*H2(p[2]), Hxx4(p[1])*H2(p[2]),
          Hxx3(p[1])*H3(p[2]), Hxx4(p[1])*H3(p[2]), Hxx3(p[1])*H4(p[2]), Hxx4(p[1])*H4(p[2]),
          Hxx1(p[1])*H3(p[2]), Hxx2(p[1])*H3(p[2]), Hxx1(p[1])*H4(p[2]), Hxx2(p[1])*H4(p[2])]'

Hyy(p) = [H1(p[1])*Hxx1(p[2]), H2(p[1])*Hxx1(p[2]), H1(p[1])*Hxx2(p[2]), H2(p[1])*Hxx2(p[2]),
          H3(p[1])*Hxx1(p[2]), H4(p[1])*Hxx1(p[2]), H3(p[1])*Hxx2(p[2]), H4(p[1])*Hxx2(p[2]),
          H3(p[1])*Hxx3(p[2]), H4(p[1])*Hxx3(p[2]), H3(p[1])*Hxx4(p[2]), H4(p[1])*Hxx4(p[2]),
          H1(p[1])*Hxx3(p[2]), H2(p[1])*Hxx3(p[2]), H1(p[1])*Hxx4(p[2]), H2(p[1])*Hxx4(p[2])]'

const Hi = vcat(H.(points)...)
const Hxxi = vcat(Hxx.(points)...)
const Hyyi = vcat(Hyy.(points)...)

h(x::Real) = [H1(x) H2(x) H3(x) H4(x)]
hx(x::Real) = [Hx1(x) Hx2(x) Hx3(x) Hx4(x)]
h(x::Vector) = vcat(h.(x)...)
hx(x::Vector) = vcat(hx.(x)...)

const hi = h(P)
const hxi = hx(P)
const Wthi = W' * hi

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

interpolate(data, mesh) = begin
    f(x, y) = begin
        ne = mesh["ne"]
        ine = 0
        for i in 1:ne
            indice = mesh["elnodes"][:, i]
            if (
                    mesh["nodes"][1, indice[1]] <= x && 
                    mesh["nodes"][1, indice[2]] >= x && 
                    mesh["nodes"][2, indice[1]] <= y &&
                    mesh["nodes"][2, indice[3]] >= y
                    )
                ine = i
            end
        end
        nodes = mesh["nodes"][:, mesh["elnodes"][:, ine]]
        Δx = nodes[1, 2] - nodes[1, 1]
        Δy = nodes[2, 4] - nodes[2, 1]
        ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]
        ξ = (x - 0.5*(nodes[1, 2] + nodes[1, 1])) * 2 / Δx
        η = (y - 0.5*(nodes[2, 4] + nodes[2, 1])) * 2 / Δy
        H([ξ, η]) ⋅ (data[:, mesh["elnodes"][:, ine]] .* ratio)
    end
end
