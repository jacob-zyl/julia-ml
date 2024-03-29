using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using CairoMakie
using Zygote: dropgrad, ignore, Buffer, jacobian, hessian
using ForwardDiff: derivative
using JLD
using Quadrature, HCubature

const NK = 4
const oneNK = 2

gen(ng = 7) = begin
    # fem_dict = load("prob.jld")
    nn = (ng + 1)^2
    ne = ng^2

    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    side = ng^-1 * (0:ng) |> collect
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(4, nn)
    data[1, :] = ones(nn)

    data[1, upper_wall] = nodes[1, upper_wall] .|> sinpi
    # data[3, upper_wall] = (nodes[1, upper_wall] .|> sinpi) .* (pi / tanh(pi))
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

    u = oneHi * f
    uinit = oneHi * finit
    
    get_global_xy(p) = (p[1] * 0.5Δx + 0.5nodes[1, 2] + 0.5nodes[1, 1], p[2] * 0.5Δy + 0.5nodes[2, 4] + 0.5nodes[2, 1])

    ## coefficients below are from element governing equation
    
    uxx = oneHxxi * f * 4.0Δx^-2
    uyy = oneHyyi * f * 4.0Δy^-2
    
    ux = oneHxxi * f * 2.0Δx^-1
    uy = oneHyyi * f * 2.0Δy^-1

    source_term = @. fs(get_global_xy(onepoints))

    nu = 1.0
    rdt = 1.0 / dt
    #r = @. 0.5nu * (ux^2 + uy^2) - u * ( (-0.5u + uinit) * rdt + source_term)
    
    r = @. ( (u - uinit) / dt - nu * (uxx + uyy) - source_term )^2

    oneweights ⋅ r * Δx * Δy * 0.25
end

element_loss(nodes, data, init, dt) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]
    get_global_xy(p) = (p[1] * 0.5Δx + 0.5nodes[1, 2] + 0.5nodes[1, 1], 
        p[2] * 0.5Δy + 0.5nodes[2, 4] + 0.5nodes[2, 1])

    f = @views (data .* ratio)[:]
    finit = @views (init .* ratio)[:]

    u = wHi ⋅ f
    uinit = wHi ⋅ finit
    # damn... magic numbers
    Nindex = [13, 14, 9, 10]
    Sindex = [1, 2, 5, 6]
    Eindex = [5, 7, 9, 11]
    Windex = [1, 3, 13, 15]

    uN = NHi * finit
    uS = SHi * finit
    uW = WHi * finit
    uE = EHi * finit
    
    fN = W ⋅ (uN .* us.(get_global_xy.(Npoints)))
    fS = W ⋅ (uS .* us.(get_global_xy.(Spoints)))
    fW = W ⋅ (uW .* us.(get_global_xy.(Wpoints)))
    fE = W ⋅ (uE .* us.(get_global_xy.(Epoints)))
        
    r = ((u - uinit) / dt + fN * 0.5Δx + fE * 0.5Δy + fS * 0.5Δx + fW * 0.5Δy)^2
end

train(ng=10) = begin
    gen(ng)
    mesh = load("cd/mesh.jld")
    data = load("cd/data.jld", "data")
    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    opt_f_diffusive = OptimizationFunction(loss_diffusive, GalacticOptim.AutoZygote())

    dt = 0.02
    iteration = 0
    maxiteration = 20
    while true
        
        prob = OptimizationProblem(opt_f, data, (dt, mesh, data))
        sol = solve(prob, BFGS())
        @printf "%.3e\t" sol.minimum
        data = sol.minimizer

        prob_diffusive = OptimizationProblem(opt_f_diffusive, data, (dt, mesh, data))
        sol_diffusive = solve(prob_diffusive, BFGS(), maxiters=100)
        @printf "%.3e\n" sol_diffusive.minimum
        data = sol_diffusive.minimizer
        
        iteration += 1
        @save "cd/split_"*(@sprintf "%02i" ng)*"_result"*(@sprintf "%04i" iteration)*".jld" data mesh
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
        sum += element_loss(elnode, eldata, elinit, dt)
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
f_test(x) = [f_exact(x[1], x[2]), fx_exact(x[1], x[2]), fy_exact(x[1], x[2]), fxy_exact(x[1], x[2])]
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