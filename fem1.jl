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
const BATCH_SIZE = 200
### Dependent constants
### END

function build_my_model()
    acFun(x::Float32) = min(relu(x+one(x)), relu(one(x)-x))
    third = 1.0f0 / 3.0f0
    grid = [0.0f0, third, 2.0f0*third, 1.0f0]
    ϕ(θ, x) = begin
        reshape(θ, 1, :) * acFun.(3.0f0 * (x .- grid))
    end
    θ = zeros(Float32, 4) #for example
    (ϕ, θ)
end

function get_loss(ϕ)
    loss(θ, p) = begin
        u1, u2, u3, u4 = θ
        (1/27)*(9 - 12*u2 + 168*u2^2 - 6*u3 -
   159*u2*u3 + 168*u3^2 + 53*u4 -
   159*u3*u4 + 111*u4^2)
    end
end

# function train(; kwargs...)
#     ϕ, f, g, h, b, c, a = build_my_model()
#     train(ϕ, [f g h b c a]; kwargs...)
# end

function train(; kwargs...)
    ϕ, f = build_my_model()
    train(ϕ, f; kwargs...)
end

function train(ϕ, Θ::Array; optimizer=ADAM(), maxiters=500)
    domain = 0f0
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
    f_exact(x) = 1.0f0 - x - exp(-x)
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
