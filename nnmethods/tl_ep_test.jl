using LinearAlgebra, Statistics, StaticArrays
using Flux, DiffEqFlux
using GalacticOptim, Optim
using Plots
pyplot()
theme(:vibrant)

push!(LOAD_PATH, pwd())
using Utils

### CONSTANTS
###
### Independent constants
const DIM = 1
const DIM_OUT = 1
const HIDDEN = 16
const BATCH_SIZE = 30
### Dependent constants
### END


function get_training_set()
    get_domain(DIM, BATCH_SIZE)
    #return [0.1f0 0.2f0 0.3f0]
end

function build_my_model(hidden=HIDDEN)
    n = range(1.0f0, step=1.0f0, length=hidden) |> collect
    base_sin(x::Matrix{Float32}) = begin
        @. sin((n * x))
    end

    base_x(x::Matrix{Float32}) = begin
        @. exp(n * log(x))
    end

    base_cos(x::Matrix{Float32}) = begin
        @. cos(n * x)
    end

    ϕ(θ, x) = begin
        ## split the parameters first
        θ' * [zero(x) .+ 1.0f0; base_x(x); base_sin(x); base_cos(x)]
    end
    θ1 = zeros(Float32, 3hidden+1)
    θ2 = zeros(Float32, 3hidden+1)
    (ϕ, θ1, θ2)
end

function get_loss(ϕ)
    # reduce_func = cliff ∘ abs2
    reduce_func = abs2
    loss(θ, p) = begin
        u = θ * [1.0f0; 0.0f0]
        v = θ * [0.0f0; 1.0f0]
        domain = p
        Du = D(ϕ, u, domain)
        Dv = D(ϕ, v, domain)

        +(
            mean(reduce_func, ϕ(u, [1.0f0 0.0f0]) - [0.0f0 0.0f0]),
            mean(reduce_func, Dv + ϕ(u, domain) + domain),
            mean(reduce_func, Du - ϕ(v, domain)),
        )
    end
end

function train(; kwargs...)
    ϕ, u, v = build_my_model()
    train(ϕ, [u v]; kwargs...)
end

function train(ϕ, Θ::Array; optimizer=ADAM(), maxiters=500)
    domain = get_training_set()
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

function show_results(ϕ, sol)
    θ = sol.minimizer * [1.0f0; 0.0f0]
    x_test = reshape(range(0.0f0, 1.0f0, length=77) |> collect, 1, :)
    f_exact(x) = sin(x)/sin(1.0f0) - x
    p = plot(x_test', f_exact.(x_test'))
    plot!(p, x_test', ϕ(θ, x_test)',
          linestyle=:dot)
    q = plot(x_test', f_exact.(x_test') - ϕ(θ, x_test)')
    plot(p, q)
end
