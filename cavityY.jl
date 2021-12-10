push!(LOAD_PATH, pwd())
using DEUtils
using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using CairoMakie
using Zygote: dropgrad, ignore, Buffer, jacobian, hessian
using ForwardDiff: derivative
using JLD

const DIR = "driven_cavityY/"

gen(;ng=20, dir = DIR) = begin
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
    
    wall = 1:nn |> collect
    filter!(e -> e ∉ inner, wall)

    @save dir*"mesh.jld" ng ne nn elnodes nodes all_walls inner wall
    @save dir*"data.jld" data
end

train(data_init, mesh_init; 
      task="re100_result", dir=DIR, nu = 0.01) = begin
    data = data_init
    mesh = mesh_init
    ng = size(data, 2) |> sqrt |> Int |> x -> x - 1
    
    opt_f2 = OptimizationFunction(loss2, GalacticOptim.AutoZygote())
    opt_f3 = OptimizationFunction(loss3, GalacticOptim.AutoZygote())

    
for iteration = 1:1000

    prob2 = OptimizationProblem(opt_f2, data, (nu, mesh))
    sol2 = solve(prob2, ConjugateGradient())
    @printf "%.2e\t" sol2.minimum

    err = sum(abs2, sol2.minimizer[1, :] - data[1, :])
    data = sol2.minimizer

    prob3 = OptimizationProblem(opt_f3, data, (nu, mesh))
    sol3 = solve(prob3, ConjugateGradient())
    @printf "%.2e\t" sol3.minimum

    
    @printf "%.2e\n" err
    data = sol3.minimizer

    file_name = (dir * task * "_grid" * (@sprintf "%02i" ng) *
                 "_" * (@sprintf "%04i" iteration) * ".jld")
    @save file_name data mesh

    if err < 5e-7 && iteration > 1
        break
    end
end
end
train(;ng=10, task="re100_result", dir=DIR, nu = 0.01) = begin
    gen(ng = ng, dir = dir)
    mesh = load(dir*"mesh.jld")
    data = load(dir*"data.jld", "data")
    train(data, mesh; task = task, dir = dir, nu = nu)
end
train(jldfile; task="re100_new", dir=DIR, nu = 0.01) = begin
    mesh = load(jldfile, "mesh")
    data = load(jldfile, "data")
    train(data, mesh; task = task, dir = dir, nu = nu)
end


loss2(data, fem_dict) = begin
    nu, mesh = fem_dict
    ne = mesh["ne"]
    nodes = mesh["nodes"]
    elnodes = mesh["elnodes"]
    wall = mesh["wall"]

    buf = Buffer(data)
    
    buf[:, :] = data[:, :]
    # keep all stream function as constants
    buf[1:4, wall] = dropgrad(data[1:4, wall])
    buf[5:8, :] = dropgrad(data[5:8, :])
    
    data = copy(buf)
    
    sum = 0
    for iters in 1:ne
        indice = elnodes[:, iters]
        elnode = @views nodes[:, indice]
        eldata = @views data[:, indice]
        sum += element_loss2(elnode, eldata, nu)
    end
    sum
end
element_loss2(nodes, data, nu) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]

    ψdata     = @views (data[1:4, :] .* ratio)[:]
    ωdata     = @views (data[5:8, :] .* ratio)[:]

    ## coefficients below are from element governing equation
    #ωxx     = Hxxi * ωdata * 4 * Δx^-2
    #ωyy     = Hyyi * ωdata * 4 * Δy^-2
    #Δω = ωxx + ωyy
    laplacian = DEUtils.HXX_2D * 4Δx^-2 + DEUtils.HYY_2D * 4Δy^-2
    Δω = laplacian * ωdata
    
    ψx  = DEUtils.HX_2D  * ψdata * 2 * Δx^-1
    ψy  = DEUtils.HY_2D  * ψdata * 2 * Δy^-1
    
    ωx  = DEUtils.HX_2D  * ωdata * 2 * Δx^-1
    ωy  = DEUtils.HY_2D  * ωdata * 2 * Δy^-1

    res = @. (ψy * ωx - ψx * ωy - nu * Δω)^2
    DEUtils.WEIGHTS_2D ⋅ res * Δx * Δy * 0.25
end
loss3(data, fem_dict) = begin
    _, mesh = fem_dict
    ne = mesh["ne"]
    nodes = mesh["nodes"]
    elnodes = mesh["elnodes"]
    inner = mesh["inner"]
    wall = mesh["wall"]

    buf = Buffer(data)
    buf[:, :] = data[:, :]
    
    buf[1:4, :] = dropgrad(data[1:4, :])
    
    #buf[5:8, inner] = dropgrad(data[5:8, inner])
    
    data = copy(buf)

    sum = 0
    for iters in 1:ne
        indice = elnodes[:, iters]
        elnode = @views nodes[:, indice]
        eldata = @views data[:, indice]
        sum += element_loss3(elnode, eldata)
    end
    sum
end
element_loss3(nodes, data) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]

    ψdata     = @views data[1:4, :] .* ratio |> vec
    ωdata     = @views data[5:8, :] .* ratio |> vec
    
    ω = DEUtils.H_2D_LESS * ωdata
    ψ = DEUtils.H_2D_LESS * ψdata

    ## coefficients below are from element governing equation
    #ψxx     = oneHxxi * ψdata * 4 * Δx^-2
    #ψyy     = oneHyyi * ψdata * 4 * Δy^-2
    #Δψ = ψxx + ψyy
    
    laplacian = DEUtils.HXX_2D_LESS * 4Δx^-2 + DEUtils.HYY_2D_LESS * 4Δy^-2
    Δψ = laplacian * ψdata

    residual1 = @. (ω + Δψ)^2
    DEUtils.WEIGHTS_2D_LESS ⋅ (residual1)
end


###### Below are some utility code #####


walls(NG) = begin
    upper_wall = ((NG+1)*NG+1):(NG+1)^2
    lower_wall = 1:(NG+1)
    left_wall = 1:(NG+1):(NG*(NG+1)+1)
    right_wall = (NG+1):(NG+1):(NG+1)^2
    (upper_wall, lower_wall, left_wall, right_wall)
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

show_map(data, mesh; levels=10) = begin
    ng = sqrt(length(size(data, 2))) - 1 |> Integer
    xs = 0:0.001:1
    ys = 0:0.001:1
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, interpolate(data, mesh), colormap=:bwr)
    ct = contour!(ax, xs, ys, 
		  [interpolate(data, mesh)(x, y) for x in xs, y in ys],
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
        DEUtils.hermite2d((ξ, η)) ⋅ (data[:, mesh["elnodes"][:, ine]] .* ratio)
    end
end