using LinearAlgebra, Statistics, StaticArrays
using Flux, DiffEqFlux
using GalacticOptim, Optim
using Plots
push!(LOAD_PATH, pwd())
using Utils

pyplot()
theme(:vibrant)

### CONSTANTS
###
### Independent constants
const DIM = 1
const DIM_OUT = 1
const BATCH_SIZE = 200
### Dependent constants
const LENGTH = 10
### END

function acFun(x)
    # ## version 1 (this is not vectorized)
    # if x > -1.0f0 && x <= 0.0f0
    #     return 1.0f0 + x
    # elseif x > 0.0f0 && x < 1.0f0
    #     return 1.0f0 - x
    # else
    #     return 0.0f0
    # end

    # ## version 2 (this is not vectorized)
    # p = 0.0f0
    # if x > -1.0f0 && x <= 0.0f0
    #     p = 1.0f0 + x
    # elseif x > 0.0f0 && x < 1.0f0
    #     p = 1.0f0 - x
    # end
    # p

    ## version 3
    min(relu(x+one(x)), relu(one(x)-x))

    # ## version 4
    # max(one(x) - abs(x), zero(x))

end

function build_my_model()
    GRID = range(0.0f0, 1.0f0, length=LENGTH) |> collect
    DX = GRID[2] - GRID[1]
    rDX = 1.0f0 / DX
    ϕ(θ::Vector{Float32}, x::Matrix{Float32}) = begin
        reshape(θ, 1, :) * acFun.(rDX .* (x .- GRID))
    end
    θ1 = zeros(Float32, LENGTH)
    (ϕ, θ1)
end

function get_loss(ϕ)

    loss(θ, domain) = begin
        f = θ
        # reduce_func = cliff ∘ abs2
        reduce_func = abs2

        # eq_res = D(ϕ, f, domain) - ϕ(f, domain)
        # bd_res = ϕ(f, zeros(Float32, 1, 1)) .- ones(Float32, 1, 1)
        # +(
        #     mean(reduce_func, eq_res),
        #     mean(reduce_func, bd_res))
        res = ϕ(f, domain) - exp.(domain)
        mean(reduce_func, res)
    end
    # loss_hard(θ, p) = begin
    #     r = loss(θ, p)
    #     cliff(r)
    # end
end

function train(ϕ, Θ::Array; optimizer=ADAM(), maxiters=500)
    # domain = reshape(
    #     range(0.0f0, stop=1.0f0, length=BATCH_SIZE) |> collect,
    #     1, :)
    domain = rand(Float32, (1, BATCH_SIZE))
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

function show_results(ϕ, sol)
    f_exact(x) = exp(x)
    x_test = reshape(range(0f0, 1f0, length=200), 1, :) |> collect
    p = plot(x_test', ϕ(sol.minimizer, x_test)',
             label=" Simulation Result")
    plot!(p, x_test', f_exact.(x_test'),
          linestyle=:dot,
          label=" Analytical Solution")
    q = plot(x_test', f_exact.(x_test') - ϕ(sol.minimizer, x_test)',
             label=" Pointwise Error")
    plot(p, q)
end
