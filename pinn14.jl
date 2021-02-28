using LinearAlgebra, Statistics, StaticArrays
using Flux, DiffEqFlux, Zygote
using GalacticOptim, Optim, Quadrature
using Printf
using Plots
plotlyjs()
theme(:vibrant)

### CONSTANTS
###
### Independent constants
const DIM = 2
const HIDDEN = 20
const BATCH_SIZE = 100
### Dependent constants
const BDSIZE = BATCH_SIZE |> sqrt |> ceil |> Int # boundary data batch size
const P_SIZE = DIM * HIDDEN + HIDDEN + HIDDEN * 1 + 1
const P1 = DIM * HIDDEN
const P2 = HIDDEN
const P3 = HIDDEN * 1
const P4 = 1
const PJ11 = diagm(P1, P_SIZE, 0 => ones(Float32, P1))
const PJ12 = diagm(P2, P_SIZE, P1 => ones(Float32, P2))
const PJ21 = diagm(P3, P_SIZE, P1 + P2 => ones(Float32, P3))
const PJ22 = diagm(P4, P_SIZE, P1 + P2 + P3 => ones(Float32, P4))
### END

function show_results(ϕ, sol)
    x_test = range(0, 1, length=200)
    y_test = range(0, 1, length=200)
    func(x) = ϕ(sol.minimizer[:, 1], reshape(x, 2, 1))[1]
    graph_f = vcat.(x_test', y_test) .|> func
    p = heatmap(x_test, y_test, graph_f, aspect_ratio=:equal)
    exact(ξ) = begin
        x, y = ξ
        sin(pi * x) * sinh(pi * y) / sinh(pi)
    end
    f_error(x, p) = func(x) - exact(x) |> abs2
    qprob = QuadratureProblem(f_error, zeros(2), ones(2))
    solution = solve(qprob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
    contour!(p, x_test, y_test, graph_f,
             line=(:black),
             contour_labels=true,
             levels=10,
             colorbar_entry=false,
             xlims=(0, 1),
             ylims=(0, 1))
    annotate!(p, 0.5, 0.2, "Error = $(@sprintf("%.1e", sqrt(solution.u[1])))")
    #q = contourf(x_test, y_test, (x, y) -> f([x; y], sol.minimizer)[1])
    (sqrt(solution.u[1]), p)
end

function get_domain(;show=false)
    domain = rand(Float32, DIM, BATCH_SIZE)
    if show
        p = scatter(zip(domain[1, :], domain[2, :]) |> collect,
                    aspect_ratio=:equal,
                    framestyle=:box,
                    ticks=0:0.2:1,
                    xlims=(0, 1),
                    ylims=(0, 1))
        return (p, domain)
    end
    domain
end

function build_model_fast()
    network = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
    θ1 = initial_params(network)
    θ2 = initial_params(network)
    θ3 = initial_params(network)
    (ϕ1, θ1, θ2, θ3)
end

function build_my_model()
    ϕ(θ::Array{Float32, 1}, x::Array{Float32, 2})::Array{Float32, 2} = begin
        W1 = reshape(PJ11::Array{Float32, 2} * θ, HIDDEN, DIM)
        b1 = reshape(PJ12::Array{Float32, 2} * θ, HIDDEN, 1)
        W2 = reshape(PJ21::Array{Float32, 2} * θ, 1, HIDDEN)
        b2 = reshape(PJ22::Array{Float32, 2} * θ, 1, 1)

        W2 * (tanh.(W1 * x .+ b1)) .+ b2
    end
    _, θ1, θ2, θ3 = build_model_fast()
    (ϕ, θ1, θ2, θ3)
end

"""
f is f(x, θ)
"""
function get_loss(ϕ::H) where{H}

    D(ϕ) = (f::Array{Float32, 1}, x::Array{Float32, 2}) -> pullback(ξ -> ϕ(f, ξ), x)[2](ones(Float32, 1, size(x, 2)))[1]
    #Dx(ϕ) = (f::Array{Float32, 1}, x::Array{Float32, 2}) -> [1.0f0 0.0f0] * pullback(ξ -> ϕ(f, ξ), x)[2](ones(Float32, 1, size(x, 2)))[1]
    #Dy(ϕ) = (f::Array{Float32, 1}, x::Array{Float32, 2}) -> [0.0f0 1.0f0] * pullback(ξ -> ϕ(f, ξ), x)[2](ones(Float32, 1, size(x, 2)))[1]

    cliff(x; a=1.0f-3, b=1.0f5) = x + log1p(b * x) * a # This is magic (number)
    split(phi::T) where{T} = (
        (f::Array{Float32, 1}, x::Array{Float32, 2}) -> [1.0f0 0.0f0] * phi(f, x)::Array{Float32, 2},
        (f::Array{Float32, 1}, x::Array{Float32, 2}) -> [0.0f0 1.0f0] * phi(f, x)::Array{Float32, 2})

    ∂x, ∂y = split(D(ϕ))
    #∂x = Dx(ϕ)
    #∂y = Dy(ϕ)

    points = range(0.0f0, 1.0f0, length = BDSIZE)'
    bd_1 = [zeros(Float32, 1, BDSIZE); points]
    bd_2 = [ones(Float32, 1, BDSIZE); points]
    bd_3 = [points; zeros(Float32, 1, BDSIZE)]
    bd_4 = [points; ones(Float32, 1, BDSIZE)]
    f_1 = zeros(Float32, 1, BDSIZE)
    f_2 = zeros(Float32, 1, BDSIZE)
    f_3 = zeros(Float32, 1, BDSIZE)
    f_4 = @. sin(pi * points) * pi / tanh(pi)

    loss(θ::Array{Float32, 2}, pde_domain::Array{Float32, 2}) = begin
        f = θ * [1.0f0; 0.0f0; 0.0f0]
        g = θ * [0.0f0; 1.0f0; 0.0f0]
        h = θ * [0.0f0; 0.0f0; 1.0f0]
        reduce_func = cliff ∘ abs2

        eq_res_1 = ∂x(g, pde_domain) + ∂y(h, pde_domain)
        eq_res_2 = ∂x(f, pde_domain) - ϕ(g, pde_domain)
        eq_res_3 = ∂y(f, pde_domain) - ϕ(h, pde_domain)
        bd_res_1 = ϕ(f, bd_1) - f_1
        bd_res_2 = ϕ(f, bd_2) - f_2
        bd_res_3 = ϕ(f, bd_3) - f_3
        bd_res_4 = ∂y(f, bd_4) - f_4

        +(
            mean(reduce_func, eq_res_1),
            mean(reduce_func, eq_res_2),
            mean(reduce_func, eq_res_3),
            mean(reduce_func, bd_res_1),
            mean(reduce_func, bd_res_2),
            mean(reduce_func, bd_res_3),
            mean(reduce_func, bd_res_4))
    end
    # loss_hard(θ, p) = begin
    #     r = loss(θ, p)
    #     cliff(r)
    # end
end

function train(ϕ, Θ::Array; optimizer=BFGS(), maxiters=1000)
    pde_domain = get_domain()
    opt_f = OptimizationFunction(get_loss(ϕ), GalacticOptim.AutoZygote())
    prob = OptimizationProblem(opt_f, Θ, pde_domain)
    sol = solve(prob, optimizer; maxiters=maxiters)
    (ϕ, sol)
end

function train(ϕ, sol::Optim.MultivariateOptimizationResults; kwargs...)
    train(ϕ, sol.minimizer; kwargs...)
end

function train(ϕ, sol::GalacticOptim.OptimizationSolution; kwargs...)
    train(ϕ, sol.minimizer; kwargs...)
end

function train(; kwargs...)
    ϕ, f, g, h = build_my_model()
    train(ϕ, [f g h]; kwargs...)
end

#f, s = train();
