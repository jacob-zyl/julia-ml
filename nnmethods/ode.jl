using LinearAlgebra, Statistics, StaticArrays
using Flux, DiffEqFlux
using GalacticOptim, Optim
using Plots
push!(LOAD_PATH, pwd())
using Utils
using ConstantsODE

pyplot()
theme(:vibrant)

function build_model_fast()
    network = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
    θ1 = initial_params(network)
    (network, θ1)
end

function build_my_model()
    ϕ(θ, x) = begin
        W1 = reshape(PJ11 * θ, HIDDEN, DIM)
        b1 = reshape(PJ12 * θ, HIDDEN, 1)
        W2 = reshape(PJ21 * θ, 1, HIDDEN)
        b2 = reshape(PJ22 * θ, 1, 1)

        W2 * (tanh.(W1 * x .+ b1)) .+ b2
    end
    _, θ1 = build_model_fast()
    (ϕ, θ1)
end

function get_loss(ϕ)

    loss(θ, domain) = begin
        f = θ
        reduce_func = cliff ∘ abs2

        fx = D(ϕ, f, domain)

        eq_res = D(ϕ, f, domain) - ϕ(f, domain)

        bd_res = ϕ(f, zeros(Float32, 1, 1)) .- ones(Float32, 1, 1)

        +(
            mean(reduce_func, eq_res),
            mean(reduce_func, bd_res))
    end
    # loss_hard(θ, p) = begin
    #     r = loss(θ, p)
    #     cliff(r)
    # end
end

function train(ϕ, Θ::Array; optimizer=BFGS(), maxiters=500)
    domain = reshape(range(0.0f0, stop=1.0f0, length=100) |> collect, 1, :)
    opt_f = OptimizationFunction(get_loss(ϕ), GalacticOptim.AutoZygote())
    prob = OptimizationProblem(opt_f, Θ, domain)
    sol = solve(prob, optimizer; maxiters=maxiters)
    (ϕ, sol)
end

function train(ϕ, sol::Optim.MultivariateOptimizationResults; kwargs...)
    train(ϕ, sol.minimizer; kwargs...)
end

function train(ϕ, sol::SciMLBase.OptimizationSolution; kwargs...)
    train(ϕ, sol.minimizer; kwargs...)
end

function train(; kwargs...)
    ϕ, f = build_my_model()
    train(ϕ, f; kwargs...)
end

# ## Profiling
# using Profile
# f, s = train()

# Profile.clear()
# @profile train()
# Juno.profiler(; C=true)
# @profiler train() combine=true
