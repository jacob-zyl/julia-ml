using LinearAlgebra, Statistics, StaticArrays
using Flux, DiffEqFlux
using GalacticOptim, Optim
using Plots
using Printf
pyplot()
theme(:wong)

push!(LOAD_PATH, pwd())
using Utils

function get_training_set()
    return 0.0f0
end

function build_my_model()
    ϕ(θ, x) = begin
        a1 = θ[1]
        a2 = θ[2]
        @. x * (1.0f0 - x) * (a1 + a2 * x)
    end
    θ = zeros(Float32, 1, 2)
    (ϕ, θ)
end

function get_loss(ϕ)
    loss(θ, p) = begin
        a1 = θ[1]
        a2 = θ[2]
        E2 = a1^2 * 101.0f0 / 30.0f0 + a1*a2 * 101.f0 / 30.0f0 - a1 * 11.0f0 / 6.0f0 + a2^2 * 131.0f0 / 35.0f0 - a2 * 19.0f0 / 10.0f0 + 1.0f0 / 3.0f0
        E2
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

function show_results(ϕ, θ::Array)
    f_exact(x) = sin(x) / sin(1) - x
    x_test = range(0.0f0, 1.0f0, length=133)
    y_test = f_exact.(x_test)
    y_pred = ϕ(θ, x_test') |> vec
    p = plot(x_test, y_test)
    plot!(p, x_test, y_pred,
          linestyle=:dot)
    annotate!(p, 0.5, 0.01,
              "Error = $(@sprintf("%.1e", sqrt(mean(abs2, y_pred - y_test))))")
    p
end

function show_results(ϕ, sol::Optim.MultivariateOptimizationResults)
    show_results(ϕ, sol.minimizer)
end

function show_results(ϕ, sol::SciMLBase.OptimizationSolution)
    show_results(ϕ, sol.minimizer)
end
