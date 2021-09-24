using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf
using GLMakie, CairoMakie
using Zygote: dropgrad, ignore, Buffer, jacobian, hessian
using ForwardDiff: derivative
using JLD
GLMakie.activate!()

const NK = 4

gen() = begin
    # fem_dict = load("prob.jld")
    ng = 10
    nn = (ng + 1)^2
    ne = ng^2

    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    side = ng^-1 * (0:ng) |> collect
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(8, nn)

    data[3, upper_wall] = nodes[1, upper_wall] .|> (x -> 16x^2*(1-x)^2)
    # data[3, upper_wall] = (nodes[1, upper_wall] .|> sinpi) .* (pi / tanh(pi))
    # data[1, lower_wall] = nodes[1, lower_wall] .|> sinpi
    # data[1, left_wall] = -nodes[2, left_wall] .|> sinpi
    # data[1, right_wall] = 1 .- nodes[2, right_wall] .|> sinpi

    elnodes = hcat(map(p -> e2nvec(p, ng), 1:ne)...)

    @save "mesh.jld" ng ne nn elnodes nodes
    @save "data.jld" data
end

train(ng) = begin
    gen(ng)
    mesh = load("mesh.jld")
    data = load("data.jld", "data")
    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())

    dt = 0.1
    time = 0.0
    for iteration in 1:100
        prob = OptimizationProblem(opt_f, data, (dt, mesh, data))
        sol = solve(prob, ConjugateGradient())
        data = sol.minimizer
        time += dt
        @save "driven_cavity_data"*(@sprintf "%04i" iteration)*".jld" data time
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
        elinit = @views data_init[:, indice]
        sum += element_loss(elnode, eldata, elinit, dt)
    end
    sum
end

element_loss(nodes, data, init, dt) = begin
    Δx = nodes[1, 2] - nodes[1, 1]
    Δy = nodes[2, 4] - nodes[2, 1]
    ratio = Float64[1, 0.5Δx, 0.5Δy, 0.25Δx*Δy]

    ψdata     = @views data[1:4, :] .* ratio
    ωdata     = @views data[5:8, :] .* ratio
    initψdata = @views init[1:4, :] .* ratio
    initωdata = @views init[5:8, :] .* ratio

    ψ = Hi * vec(ψdata)
    ω = Hi * vec(ωdata)

    initψ = Hi * vec(initψdata)
    initω = Hi * vec(initωdata)

    # coefficients below are from element governing equation
    ωxx     = Hxxi * vec(ωdata)     * 4 * Δx^-2
    initωxx = Hxxi * vec(initωdata) * 4 * Δx^-2

    ωyy     = Hyyi * vec(ωdata)     * 4 * Δy^-2
    initωyy = Hyyi * vec(initωdata) * 4 * Δy^-2

    ψxx     = Hxxi * vec(ψdata)     * 4 * Δx^-2
    initψxx = Hxxi * vec(initψdata) * 4 * Δx^-2

    ψyy     = Hyyi * vec(ψdata)     * 4 * Δy^-2
    initψyy = Hyyi * vec(initψdata) * 4 * Δy^-2

    ωx      = Hxi * vec(ωdata)      * 2 * Δx^-1
    initωx  = Hxi * vec(initωdata)  * 2 * Δx^-1

    ωy      = Hyi * vec(ωdata)      * 2 * Δy^-1
    initωy  = Hyi * vec(initωdata)  * 2 * Δy^-1

    ψx      = Hxi * vec(ψdata)      * 2 * Δx^-1
    initψx  = Hxi * vec(initψdata)  * 2 * Δx^-1

    ψy      = Hyi * vec(ψdata)      * 2 * Δy^-1
    initψy  = Hyi * vec(initψdata)  * 2 * Δy^-1

    residual1 = @. ω + (ψxx + ψyy)
    residual2 = @. (ω - initω) / dt + 0.5 * (
        ψy * ωx - ψx * ωy - nu * (ωxx + ωyy)
    ) + 0.5 * (
        initψy * initωx - initψx * initωy - nu * (initωxx + initωyy)
    )

    weights ⋅ residual1.^2 + weights ⋅ residual2.^2
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

# correctness of code blow verified.

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

const nu = 0.01

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
