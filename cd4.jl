push!(LOAD_PATH, pwd())
using DEUtils
using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using CairoMakie
using Zygote: dropgrad, Buffer
using ForwardDiff: derivative
using JLD
using Quadrature, HCubature

gen(ng = 7) = begin
    nn = (ng + 1)^2
    ne = ng^2

    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    side = ng^-1 * (0:ng) |> collect
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(4, nn)
    data[1, :] = ones(nn)

    data[1, upper_wall] = nodes[1, upper_wall] .|> sinpi
    data[1, lower_wall] = nodes[1, lower_wall] .|> zero
    data[1, left_wall] = -nodes[2, left_wall] .|> zero
    data[1, right_wall] = 1 .- nodes[2, right_wall] .|> zero

    elnodes = hcat(map(p -> e2nvec(p, ng), 1:ne)...)

    @save "cd/mesh.jld" ng ne nn elnodes nodes
    @save "cd/data.jld" data
end

element_loss_diffusive(nodes, data, init, dt) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]
    f = @view (data .* ratio)[:]
    finit = @view (init .* ratio)[:]

    u = DEUtils.H_2D * f
    uinit = DEUtils.H_2D * finit
    
    get_global_xy(p) = (
        p[1] * 0.5Δx + 0.5nodes[1, 2] + 0.5nodes[1, 1], 
        p[2] * 0.5Δy + 0.5nodes[2, 4] + 0.5nodes[2, 1])

    ## coefficients below are from element governing equation
    
    uxx = DEUtils.HXX_2D * f * 4.0Δx^-2
    uyy = DEUtils.HYY_2D * f * 4.0Δy^-2
    
    ux = DEUtils.HX_2D * f * 2.0Δx^-1
    uy = DEUtils.HY_2D * f * 2.0Δy^-1

    source_term = @. fs(get_global_xy(DEUtils.POINTS_2D))

    nu = 1.0
    rdt = 1.0 / dt * 0.0
    
    vv = @. vs(get_global_xy(DEUtils.POINTS_2D))
    uu = @. us(get_global_xy(DEUtils.POINTS_2D))
    
    r = @. ( (u - uinit) * rdt + uu * ux + vv * uy - nu * (uxx + uyy) - source_term )^2

    DEUtils.WEIGHTS_2D ⋅ r * Δx * Δy * 0.25
end

element_loss_convective(nodes, data, init, dt) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]
    get_global_xy(p) = (
        p[1] * 0.5Δx + 0.5nodes[1, 2] + 0.5nodes[1, 1], 
        p[2] * 0.5Δy + 0.5nodes[2, 4] + 0.5nodes[2, 1])

    f = @views (data .* ratio)[:]
    finit = @views (init .* ratio)[:]

    w = DEUtils.WH_2D ⋅ f
    winit = DEUtils.WH_2D ⋅ finit

    wN = DEUtils.N_H * finit
    wS = DEUtils.S_H * finit
    wW = DEUtils.W_H * finit
    wE = DEUtils.E_H * finit
    
    fN = DEUtils.WEIGHTS_1D ⋅ (wN .* vs.(get_global_xy.(DEUtils.N_POINTS)))
    fS = DEUtils.WEIGHTS_1D ⋅ (wS .* -vs.(get_global_xy.(DEUtils.S_POINTS)))
    fW = DEUtils.WEIGHTS_1D ⋅ (wW .* -us.(get_global_xy.(DEUtils.W_POINTS)))
    fE = DEUtils.WEIGHTS_1D ⋅ (wE .* us.(get_global_xy.(DEUtils.E_POINTS)))
        
    ((w - winit) / dt + fN * 0.5Δx + fE * 0.5Δy + fS * 0.5Δx + fW * 0.5Δy)^2
end

train(ng=10) = begin
    gen(ng)
    mesh = load("cd/mesh.jld")
    data = load("cd/data.jld", "data")
    opt_f_convective = OptimizationFunction(loss_convective, GalacticOptim.AutoZygote())
    opt_f_diffusive = OptimizationFunction(loss_diffusive, GalacticOptim.AutoZygote())

    opt_f = opt_f_convective
    dt = 0.02
    iteration = 0
    maxiteration = 1
    while true

        prob = OptimizationProblem(opt_f, data, (dt, mesh, data))
        sol = solve(prob, BFGS(), maxiters=1000)
        @printf "%.3e\n" sol.minimum
        data = sol.minimizer
        
        iteration += 1
        @save "cd/steady_"*(@sprintf "%02i" ng)*"_result"*(@sprintf "%04i" iteration)*".jld" data mesh
        if iteration >= maxiteration
            println(sol.original)
            break
        end
    end
    #sol
end

loss_diffusive(data, fem_dict) = begin
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
        sum += element_loss_diffusive(elnode, eldata, elinit, dt)
    end
    sum
end

loss_convective(data, fem_dict) = begin
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
        sum += element_loss_convective(elnode, eldata, elinit, dt)
    end
    sum
end


fs(p) = begin
    x = p[1]; y = p[2]
    f1 = y^2 * cos(x) * cospi(x) * sinh(pi*y) * pi/sinh(pi)
    f2 = y^3 * sin(x) * sinpi(x) * cosh(pi*y) * pi/3sinh(pi)
    f3 = sinpi(2x) * cospi(2y) - 2sinpi(2x) * sinpi(y) * sinpi(y)
    f4 = 2y^2 * cos(x) * cospi(2x) * sinpi(y) * sinpi(y) + y^3 * sin(x) * sinpi(2x) * sinpi(2y) / 3.0
    f1 + f2 + 0.2pi^2 * f3 + 0.1pi * f4
end
us(p) = p[2]^2 * cos(p[1])
vs(p) = p[2]^3 * sin(p[1]) / 3.0

f_exact(x, y) = sinpi(x) * sinh(pi * y) / sinh(pi) - 0.1sinpi(2x) * sinpi(y) * sinpi(y)
fx_exact(x, y) = derivative(p -> f_exact(p, y), x)
fy_exact(x, y) = derivative(p -> f_exact(x, p), y)
fxy_exact(x, y) = derivative(p -> fx_exact(x, p), y)
f_test(x) = [
    f_exact(x[1], x[2]), 
    fx_exact(x[1], x[2]), 
    fy_exact(x[1], x[2]), 
    fxy_exact(x[1], x[2])]
error(data, mesh) = begin
    integrand(x, p) = (interpolate(data, mesh)(x[1], x[2]) - f_exact(x[1], x[2]))^2
    prob = QuadratureProblem(integrand, zeros(2), ones(2))
    sol = solve(prob, HCubatureJL())
    sol.u |> sqrt
end

show_map(data, mesh; levels=10) = begin
    ng = sqrt(length(size(data, 2))) - 1 |> Integer
    xs = 0:0.05:1
    ys = 0:0.05:1
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, interpolate(data, mesh), colormap=:bwr)
    ct = contour!(ax, xs, ys, [interpolate(data, mesh)(x, y) for x in xs, y in ys],
                overdraw=true, levels=levels)
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
        normalized_data = (data[:, mesh["elnodes"][:, ine]] .* ratio)[:]
        value_on_points_2d(normalized_data, (ξ, η))
    end
end
