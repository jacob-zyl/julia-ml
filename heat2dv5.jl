push!(LOAD_PATH, pwd())
using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using CairoMakie
using JLD
using Quadrature, HCubature
using DEUtils: WH_2D, N_WHY, S_WHY, W_WHX, E_WHX, value_on_points_2d
using Zygote: Buffer, dropgrad
const NK = 4

gen(ng = 7) = begin
    # fem_dict = load("prob.jld")
    nn = (ng + 1)^2
    ne = ng^2

    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    side = ng^-1 * (0:ng) |> collect
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(4, nn)
    data[1, :] = ones(nn)

    # data[1, upper_wall] = nodes[1, upper_wall] .|> sinpi
    data[3, upper_wall] = (nodes[1, upper_wall] .|> sinpi) .* (pi / tanh(pi))
    data[1, lower_wall] = nodes[1, lower_wall] .|> zero
    data[1, left_wall] = -nodes[2, left_wall] .|> zero
    data[1, right_wall] = 1 .- nodes[2, right_wall] .|> zero

    elnodes = hcat(map(p -> e2nvec(p, ng), 1:ne)...)

    @save "heat2d/mesh.jld" ng ne nn elnodes nodes
    @save "heat2d/data.jld" data
end

train(ng=7) = begin
    gen(ng)
    mesh = load("heat2d/mesh.jld")
    data = load("heat2d/data.jld", "data")
    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())

    dt = 1.0
    for iteration in 1:5
        prob = OptimizationProblem(opt_f, data, (dt, mesh, data))
        sol = solve(prob, BFGS())
        data = sol.minimizer
        @printf "%f\n" sol.minimum
        @save "heat2d/conservative_"*(@sprintf "%02i" ng)*"_result"*(@sprintf "%04i" iteration)*".jld" data mesh
    end
end

loss(data, fem_dict) = begin
    dt, mesh, init = fem_dict
    ng = mesh["ng"]
    ne = mesh["ne"]
    nodes = mesh["nodes"]
    elnodes = mesh["elnodes"]
    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    buf = Buffer(data)
    buf[:, :] = data[:, :]
    buf[1, lower_wall] = dropgrad(data[1, lower_wall])
    buf[3, upper_wall] = dropgrad(data[3, upper_wall])
    buf[1, left_wall] = dropgrad(data[1, left_wall])
    buf[1, right_wall] = dropgrad(data[1, right_wall])
    data = copy(buf)

    sum = 0
    for iters in 1:ne
        indice = elnodes[:, iters]
        elnode = @views nodes[:, indice]
        eldata = @views data[:, indice]
        elinit = @views init[:, indice]
        sum += element_loss(elnode, eldata, elinit, dt)
    end
    sum
end

element_loss(nodes, data, init, dt) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]
    f =@views (data .* ratio)[:]
    finit = @views (init .* ratio)[:]

    u = WH_2D ⋅ f
    uinit = WH_2D ⋅ finit

    fN = N_WHY ⋅ f
    fS = S_WHY ⋅ f
    fW = W_WHX ⋅ f
    fE = E_WHX ⋅ f
    
    r = 0.5(fN * Δx + fE * Δy - fS * Δx - fW * Δy) + (uinit - u) / dt

    r^2
end

f_exact(x, y) = sinpi(x) * sinh(pi * y) / sinh(pi)
fx_exact(x, y) = cospi(x) * sinh(pi * y) * pi / sinh(pi)
fy_exact(x, y) = sinpi(x) * cosh(pi * y) * pi / sinh(pi)
fxy_exact(x, y) = cospi(x) * cosh(pi * y) * pi^2 / sinh(pi)
f_test(x) = [
    f_exact(x[1], x[2]), 
    fx_exact(x[1], x[2]), 
    fy_exact(x[1], x[2]), 
    fxy_exact(x[1], x[2])]

error(result) = error(load(result, "data"), load(result, "mesh"))
error(data, mesh) = begin
    integrand(x, p) = (interpolate(data, mesh)(x[1], x[2]) - f_exact(x[1], x[2]))^2
    prob = QuadratureProblem(integrand, zeros(2), ones(2))
    sol = solve(prob, HCubatureJL())
    sol.u |> sqrt
end

using ColorSchemes
show_map(data, mesh) = begin
    ng = sqrt(length(size(data, 2))) - 1 |> Integer
    xs = 0:0.005:1
    ys = 0:0.005:1
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, interpolate(data, mesh), colormap=:heat)
    ct = contour!(ax, xs, ys, [interpolate(data, mesh)(x, y) for x in xs, y in ys], overdraw=true)
    Colorbar(fig[1, 2], hm)
    #Colorbar(fig[1, 3], ct)
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
        value_on_points_2d((data[:, mesh["elnodes"][:, ine]] .* ratio)[:], (ξ, η))
    end
end
