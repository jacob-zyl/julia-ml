using LinearAlgebra, Statistics
using GalacticOptim, Optim, Flux
using Printf
using CairoMakie
using Zygote: dropgrad, ignore, Buffer, jacobian, hessian
using ForwardDiff: derivative
using JLD

gen(ng=10) = begin
    # fem_dict = load("prob.jld")
    nn = (ng + 1)^2
    ne = ng^2

    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    side = ng^-1 * (0:ng) |> collect
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(8, nn)

    # data[3, upper_wall] .= 1.0; data[3, 1] = 0.0; data[3, end] = 0.0
    data[3, upper_wall] = nodes[1, upper_wall] .|> (x -> 16x^2*(1-x)^2)
    # data[3, upper_wall] = (nodes[1, upper_wall] .|> sinpi) .* (pi / tanh(pi))
    # data[1, lower_wall] = nodes[1, lower_wall] .|> sinpi
    # data[1, left_wall] = -nodes[2, left_wall] .|> sinpi
    # data[1, right_wall] = 1 .- nodes[2, right_wall] .|> sinpi

    elnodes = hcat(map(p -> e2nvec(p, ng), 1:ne)...)

    @save "driven_cavity/mesh.jld" ng ne nn elnodes nodes
    @save "driven_cavity/data.jld" data
end

train(data_init, mesh_init; task="re100_result", nu = 0.01, algo=BFGS()) = begin
    data  = data_init
    mesh  = mesh_init
    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob = OptimizationProblem(opt_f, data, (nu, mesh))
    sol = solve(prob, algo, maxiters=1000)
    data = sol.minimizer
    @printf "%f\n" sol.minimum
    @save "driven_cavity/"*task*".jld" data mesh
end

train(;ng=10, task="re100_result", nu = 0.01, algo=BFGS()) = begin
    gen(ng)
    mesh = load("driven_cavity/mesh.jld")
    data = load("driven_cavity/data.jld", "data")
    train(data, mesh; task = task, nu = nu, algo=algo)
end

train(jldfile; task="re100_new", nu = 0.01, algo=BFGS()) = begin
    mesh = load(jldfile, "mesh")
    data = load(jldfile, "data")
    train(data, mesh; task = task, nu = nu, algo = algo)
end

loss(data, fem_dict) = begin
    nu, mesh = fem_dict
    ng = mesh["ng"]
    ne = mesh["ne"]
    nodes = mesh["nodes"]
    elnodes = mesh["elnodes"]
    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    buf = Buffer(data)
    buf[:, :] = data[:, :]
    buf[1, lower_wall] = dropgrad(data[1, lower_wall])
    buf[1, upper_wall] = dropgrad(data[1, upper_wall])
    buf[1, left_wall]  = dropgrad(data[1, left_wall])
    buf[1, right_wall] = dropgrad(data[1, right_wall])
    buf[2, lower_wall] = dropgrad(data[2, lower_wall])
    buf[2, upper_wall] = dropgrad(data[2, upper_wall])
    buf[2, left_wall]  = dropgrad(data[2, left_wall])
    buf[2, right_wall] = dropgrad(data[2, right_wall])
    buf[3, lower_wall] = dropgrad(data[3, lower_wall])
    buf[3, upper_wall] = dropgrad(data[3, upper_wall])
    buf[3, left_wall]  = dropgrad(data[3, left_wall])
    buf[3, right_wall] = dropgrad(data[3, right_wall])
    data = copy(buf)

    sum = 0
    for iters in 1:ne
        indice = elnodes[:, iters]
        elnode = @views nodes[:, indice]
        eldata = @views data[:, indice]
        sum += element_loss(elnode, eldata, nu)
    end
    sum
end

element_loss(nodes, data, nu) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]

    ψdata     = @views data[1:4, :] .* ratio
    ωdata     = @views data[5:8, :] .* ratio

    ω = @views Hi * vec(ωdata)

    # coefficients below are from element governing equation
    ωxx     = @views Hxxi * vec(ωdata)     * 4.0 * Δx^-2
    ωyy     = @views Hyyi * vec(ωdata)     * 4.0 * Δy^-2
    ψxx     = @views Hxxi * vec(ψdata)     * 4.0 * Δx^-2
    ψyy     = @views Hyyi * vec(ψdata)     * 4.0 * Δy^-2
    ωx      = @views Hxi  * vec(ωdata)     * 2.0 * Δx^-1
    ωy      = @views Hyi  * vec(ωdata)     * 2.0 * Δy^-1
    ψx      = @views Hxi  * vec(ψdata)     * 2.0 * Δx^-1
    ψy      = @views Hyi  * vec(ψdata)     * 2.0 * Δy^-1

    residual1 = @. (ω + (ψxx + ψyy))^2
    residual2 = @. (ψy * ωx - ψx * ωy - nu * (ωxx + ωyy))^2

    weights ⋅ (residual1 + residual2)
end


walls(NG) = begin
    upper_wall = ((NG+1)*NG+1):(NG+1)^2
    lower_wall = 1:(NG+1)
    left_wall = 1:(NG+1):(NG*(NG+1)+1)
    right_wall = (NG+1):(NG+1):(NG+1)^2
    (upper_wall, lower_wall, left_wall, right_wall)
end

# correctness of code blow verified.

const NK = 4
using FastGaussQuadrature
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

Hx(p) = [Hx1(p[1])*H1(p[2]), Hx2(p[1])*H1(p[2]), Hx1(p[1])*H2(p[2]), Hx2(p[1])*H2(p[2]),
          Hx3(p[1])*H1(p[2]), Hx4(p[1])*H1(p[2]), Hx3(p[1])*H2(p[2]), Hx4(p[1])*H2(p[2]),
          Hx3(p[1])*H3(p[2]), Hx4(p[1])*H3(p[2]), Hx3(p[1])*H4(p[2]), Hx4(p[1])*H4(p[2]),
          Hx1(p[1])*H3(p[2]), Hx2(p[1])*H3(p[2]), Hx1(p[1])*H4(p[2]), Hx2(p[1])*H4(p[2])]'

Hy(p) = [H1(p[1])*Hx1(p[2]), H2(p[1])*Hx1(p[2]), H1(p[1])*Hx2(p[2]), H2(p[1])*Hx2(p[2]),
          H3(p[1])*Hx1(p[2]), H4(p[1])*Hx1(p[2]), H3(p[1])*Hx2(p[2]), H4(p[1])*Hx2(p[2]),
          H3(p[1])*Hx3(p[2]), H4(p[1])*Hx3(p[2]), H3(p[1])*Hx4(p[2]), H4(p[1])*Hx4(p[2]),
          H1(p[1])*Hx3(p[2]), H2(p[1])*Hx3(p[2]), H1(p[1])*Hx4(p[2]), H2(p[1])*Hx4(p[2])]'

const Hi = vcat(H.(points)...)
const Hxxi = vcat(Hxx.(points)...)
const Hyyi = vcat(Hyy.(points)...)
const Hxi = vcat(Hx.(points)...)
const Hyi = vcat(Hy.(points)...)

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

show_map(data, mesh; levels=10, map=true, tour=true) = begin
    ng = sqrt(length(size(data, 2))) - 1 |> Integer
    xs = 0:0.005:1
    ys = 0:0.005:1
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    if map
        hm = heatmap!(ax, xs, ys, interpolate(data, mesh), colormap=:bwr)
        Colorbar(fig[1, 2], hm)
    end
    if tour
        ct = contour!(ax, xs, ys,
                      [interpolate(data, mesh)(x, y) for x in xs, y in ys],
                      overdraw=true, levels=levels)
    end
    #Colorbar(fig[1, 3], ct)
    fig
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
