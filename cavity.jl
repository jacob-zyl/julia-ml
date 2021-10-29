using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using CairoMakie
using Zygote: dropgrad, ignore, Buffer, jacobian, hessian
using ForwardDiff: derivative
using JLD

const NK = 2
const oneNK = 2
const dir = "driven_cavity4/"

gen(ng=20) = begin
    # fem_dict = load("prob.jld")
    nn = (ng + 1)^2
    ne = ng^2

    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    side = ng^-1 * (0:ng) |> collect
    # chebside = side .|> x -> 0.5 - 0.5cospi(x)
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(8, nn)

    # data[3, upper_wall] .= 1.0
    data[3, upper_wall] = nodes[1, upper_wall] .|> (x -> 16x^2*(1-x)^2)
    data[4, upper_wall] = nodes[1, upper_wall] .|> (x -> 32x*(1-x)*(1-2x))

    elnodes = hcat(map(p -> e2nvec(p, ng), 1:ne)...)
    
    all_walls = walls(ng)
    
    inner = 1:nn |> collect
    filter!(e -> e ∉ all_walls[1], inner)
    filter!(e -> e ∉ all_walls[2], inner)
    filter!(e -> e ∉ all_walls[3], inner)
    filter!(e -> e ∉ all_walls[4], inner)

    @save dir*"mesh.jld" ng ne nn elnodes nodes all_walls inner
    @save dir*"data.jld" data
end

train(data_init, mesh_init, time_init; task="re100_result", nu = 0.01, dt = 0.1) = begin
    data = data_init
    mesh = mesh_init
    ng = size(data, 2) |> sqrt |> Int |> x -> x - 1
    upper_wall, lower_wall, left_wall, right_wall = walls(ng)
    time = time_init
    
    opt_f2 = OptimizationFunction(loss2, GalacticOptim.AutoZygote())
    opt_f3 = OptimizationFunction(loss3, GalacticOptim.AutoZygote())

    
    for iteration in 1:50
        
        prob2 = OptimizationProblem(opt_f2, data, (nu, dt, mesh, data))
        sol2 = solve(prob2, ConjugateGradient())
        @printf "%.2e\t" sol2.minimum
        data = sol2.minimizer
        
        prob3 = OptimizationProblem(opt_f3, data, (nu, dt, mesh, data))
        sol3 = solve(prob3, ConjugateGradient())
        @printf "%.2e\n" sol3.minimum
        data = sol3.minimizer
        
        time += dt
        @save dir*task*"_grid"*(@sprintf "%02i" ng)*"_"*(@sprintf "%04i" iteration)*".jld" data mesh time
    end
end
train(;ng=10, task="re100_result", nu = 0.01, dt = 0.1) = begin
    gen(ng)
    mesh = load(dir*"mesh.jld")
    data = load(dir*"data.jld", "data")
    time = 0.0
    train(data, mesh, time; task = task, nu = nu, dt = dt)
end
train(jldfile; task="re100_new", nu = 0.01, dt = 0.1) = begin
    mesh = load(jldfile, "mesh")
    data = load(jldfile, "data")
    time = load(jldfile, "time")
    train(data, mesh, time; task = task, nu = nu, dt = dt)
end


loss2(data, fem_dict) = begin
    nu, dt, mesh, data_init = fem_dict
    ng = mesh["ng"]
    ne = mesh["ne"]
    nodes = mesh["nodes"]
    elnodes = mesh["elnodes"]
    upper_wall, lower_wall, left_wall, right_wall = mesh["all_walls"]

    buf = Buffer(data)
    buf[:, :] = data[:, :]
    
    buf[1:4, :] = dropgrad(data[1:4, :])
    
    buf[5:8, lower_wall] = dropgrad(data[5:8, lower_wall])
    buf[5:8, upper_wall] = dropgrad(data[5:8, upper_wall])
    buf[5:8, left_wall]  = dropgrad(data[5:8, left_wall])
    buf[5:8, right_wall] = dropgrad(data[5:8, right_wall])
    
    data = copy(buf)
    
    sum = 0
    for iters in 1:ne
        indice = elnodes[:, iters]
        elnode = @views nodes[:, indice]
        eldata = @views data[:, indice]
        elinit = @views data_init[:, indice]
        sum += element_loss2(elnode, eldata, elinit, nu, dt)
    end
    sum
end
element_loss2(nodes, data, init, nu, dt) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]

    ψdata     = @views (data[1:4, :] .* ratio)[:]
    ωdata     = @views (data[5:8, :] .* ratio)[:]
    ψinitdata = @views (init[1:4, :] .* ratio)[:]
    ωinitdata = @views (init[5:8, :] .* ratio)[:]

    ω = oneHi * ωdata

    ωinit = oneHi * ωinitdata

    # coefficients below are from element governing equation
    ωxx     = oneHxxi * ωdata * 4 * Δx^-2
    ωyy     = oneHyyi * ωdata * 4 * Δy^-2
    
    ωx      = oneHxi  * ωdata * 2 * Δx^-1
    ωy      = oneHyi  * ωdata * 2 * Δy^-1
    
    ψinitx  = oneHxi  * ψinitdata * 2 * Δx^-1
    ψinity  = oneHyi  * ψinitdata * 2 * Δy^-1
    
    ωinitx  = oneHxi  * ωinitdata * 2 * Δx^-1
    ωinity  = oneHyi  * ωinitdata * 2 * Δy^-1

    residual2 = @. ((ω - ωinit) / dt + ψinity * ωinitx - ψinitx * ωinity - nu * (ωxx + ωyy))^2
    oneweights ⋅ residual2 * Δx * Δy * 0.25
end
loss3(data, fem_dict) = begin
    nu, dt, mesh, data_init = fem_dict
    ng = mesh["ng"]
    ne = mesh["ne"]
    nodes = mesh["nodes"]
    elnodes = mesh["elnodes"]
    inner = mesh["inner"]
    upper_wall, lower_wall, left_wall, right_wall = mesh["all_walls"]

    buf = Buffer(data)
    buf[:, :] = data[:, :]
    
    buf[1:4, lower_wall] = dropgrad(data[1:4, lower_wall])
    buf[1:4, upper_wall] = dropgrad(data[1:4, upper_wall])
    buf[1:4, left_wall]  = dropgrad(data[1:4, left_wall])
    buf[1:4, right_wall] = dropgrad(data[1:4, right_wall])
    
    buf[5:8, inner] = dropgrad(data[5:8, inner])
    
    data = copy(buf)

    sum = 0
    for iters in 1:ne
        indice = elnodes[:, iters]
        elnode = @views nodes[:, indice]
        eldata = @views data[:, indice]
        elinit = @views data_init[:, indice]
        sum += element_loss3(elnode, eldata, elinit, nu, dt)
    end
    sum
end
element_loss3(nodes, data, init, nu, dt) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]

    ψdata     = @views data[1:4, :] .* ratio
    ωdata     = @views data[5:8, :] .* ratio
    
    ω = Hi * vec(ωdata)
    ψ = Hi * vec(ψdata)

    # coefficients below are from element governing equation
    ψxx     = Hxxi * vec(ψdata)     * 4 * Δx^-2
    ψx      = Hxi  * vec(ψdata)     * 2 * Δx^-1

    ψyy     = Hyyi * vec(ψdata)     * 4 * Δy^-2
    ψy      = Hyi  * vec(ψdata)     * 2 * Δy^-1

    #residual1 = @. (ωinit + (ψxx + ψyy))^2
    #residial1 = @. (ψx^2 + ψy^2 - ψ * ωinit)
    residual1 = @. (ω + (ψxx + ψyy))^2
    weights ⋅ (residual1)
end



walls(NG) = begin
    upper_wall = ((NG+1)*NG+1):(NG+1)^2
    lower_wall = 1:(NG+1)
    left_wall = 1:(NG+1):(NG*(NG+1)+1)
    right_wall = (NG+1):(NG+1):(NG+1)^2
    (upper_wall, lower_wall, left_wall, right_wall)
end

# correctness of code blow verified.


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

h(x::Real) = [H1(x) H2(x) H3(x) H4(x)]
hx(x::Real) = [Hx1(x) Hx2(x) Hx3(x) Hx4(x)]
h(x::Vector) = vcat(h.(x)...)
hx(x::Vector) = vcat(hx.(x)...)

using FastGaussQuadrature
const P, W = gausslegendre(NK)

const points = tuple.(P', P) |> vec

const Npoints = tuple.(P, 1.0ones(NK))
const Spoints = tuple.(P, -1.0ones(NK))
const Wpoints = tuple.(-1.0ones(NK), P)
const Epoints = tuple.(1.0ones(NK), P)
    
const weights = kron(W, W)

const Hi = vcat(H.(points)...)
const Hxi = vcat(Hx.(points)...)
const Hyi = vcat(Hy.(points)...)
const Hxxi = vcat(Hxx.(points)...)
const Hyyi = vcat(Hyy.(points)...)
const wHi = weights' * Hi

const NHi = vcat(H.(Npoints)...)
const SHi = vcat(H.(Spoints)...)
const WHi = vcat(H.(Wpoints)...)
const EHi = vcat(H.(Epoints)...)

const NHxi = vcat(Hx.(Npoints)...)
const SHxi = vcat(Hx.(Spoints)...)
const WHxi = vcat(Hx.(Wpoints)...)
const EHxi = vcat(Hx.(Epoints)...)

const NHyi = vcat(Hy.(Npoints)...)
const SHyi = vcat(Hy.(Spoints)...)
const WHyi = vcat(Hy.(Wpoints)...)
const EHyi = vcat(Hy.(Epoints)...)


const NHxxi = vcat(Hxx.(Npoints)...)
const SHxxi = vcat(Hxx.(Spoints)...)
const WHxxi = vcat(Hxx.(Wpoints)...)
const EHxxi = vcat(Hxx.(Epoints)...)

const NHyyi = vcat(Hyy.(Npoints)...)
const SHyyi = vcat(Hyy.(Spoints)...)
const WHyyi = vcat(Hyy.(Wpoints)...)
const EHyyi = vcat(Hyy.(Epoints)...)

const hi = h(P)
const hxi = hx(P)
const Wthi = W' * hi


const oneP, oneW = gausslegendre(oneNK)

const onepoints = tuple.(oneP', oneP) |> vec
const oneweights = kron(oneW, oneW)

const oneHi = vcat(H.(onepoints)...)
const oneHxi = vcat(Hx.(onepoints)...)
const oneHyi = vcat(Hy.(onepoints)...)
const oneHxxi = vcat(Hxx.(onepoints)...)
const oneHyyi = vcat(Hyy.(onepoints)...)
const onewHi = oneweights' * oneHi


const onehi = h(oneP)
const onehxi = hx(oneP)
const oneWthi = oneW' * onehi


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

show_map(data, mesh; levels=10) = begin
    ng = sqrt(length(size(data, 2))) - 1 |> Integer
    xs = 0:0.001:1
    ys = 0:0.001:1
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, interpolate(data, mesh), colormap=:bwr)
    ct = contour!(ax, xs, ys, [interpolate(data, mesh)(x, y) for x in xs, y in ys],
                overdraw=true, levels=levels)
    Colorbar(fig[1, 2], hm)
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

