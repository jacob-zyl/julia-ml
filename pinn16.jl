using LinearAlgebra, Statistics, StaticArrays
using Flux, DiffEqFlux
using GalacticOptim, Optim
using Plots
plotlyjs()
theme(:vibrant)

push!(LOAD_PATH, pwd())
using Utils
using Constants

function build_model_fast()
    network = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
    θ1 = initial_params(network)
    θ2 = initial_params(network)
    θ3 = initial_params(network)
    (ϕ1, θ1, θ2, θ3)
end

function build_my_model()
    ϕ(θ, x) = begin
        W1 = reshape(PJ11 * θ, HIDDEN, DIM)
        b1 = reshape(PJ12 * θ, HIDDEN, 1)
        W2 = reshape(PJ21 * θ, 1, HIDDEN)
        b2 = reshape(PJ22 * θ, 1, 1)

        W2 * (tanh.(W1 * x .+ b1)) .+ b2
    end
    _, θ1, θ2, θ3 = build_model_fast()
    (ϕ, θ1, θ2, θ3)
end

function get_loss(ϕ)

    points = range(0.0f0, 1.0f0, length = BD_SIZE)'
    bd_1 = [zeros(Float32, 1, BD_SIZE); points]
    bd_2 = [ones(Float32, 1, BD_SIZE); points]
    bd_3 = [points; zeros(Float32, 1, BD_SIZE)]
    bd_4 = [points; ones(Float32, 1, BD_SIZE)]
    f_1 = zeros(Float32, 1, BD_SIZE)
    f_2 = zeros(Float32, 1, BD_SIZE)
    f_3 = zeros(Float32, 1, BD_SIZE)
    f_4 = @. sin(pi * points) * pi / tanh(pi)

    loss(θ, pde_domain) = begin
        f = θ * [1.0f0; 0.0f0; 0.0f0]
        g = θ * [0.0f0; 1.0f0; 0.0f0]
        h = θ * [0.0f0; 0.0f0; 1.0f0]
        reduce_func = cliff ∘ abs2

        Dxf, Dyf = getxy(D(ϕ, f, pde_domain))
        Dxg, _ = getxy(D(ϕ, g, pde_domain))
        _, Dyh = getxy(D(ϕ, h, pde_domain))
        _, Dyf2 = getxy(D(ϕ, f, bd_4))

        eq_res_1 = Dxg + Dyh
        eq_res_2 = Dxf - ϕ(g, pde_domain)
        eq_res_3 = Dyf - ϕ(h, pde_domain)

        bd_res_1 = ϕ(f, bd_1) - f_1
        bd_res_2 = ϕ(f, bd_2) - f_2
        bd_res_3 = ϕ(f, bd_3) - f_3
        bd_res_4 = Dyf2 - f_4

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

function train(ϕ, Θ::Array; optimizer=BFGS(), maxiters=500)
    pde_domain = get_domain(DIM, BATCH_SIZE)
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

# ## Profiling
# using Profile
# f, s = train()

# Profile.clear()
# @profile train()
# Juno.profiler(; C=true)
# @profiler train() combine=true
