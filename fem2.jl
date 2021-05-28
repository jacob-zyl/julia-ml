using LinearAlgebra, Statistics, StaticArrays
using Flux, DiffEqFlux
using GalacticOptim, Optim
using Plots
pyplot()
theme(:vibrant)

push!(LOAD_PATH, pwd())
using Utils

function build_my_model()
    acFun(x::Float32) = min(relu(x+one(x)), relu(one(x)-x))
    dx = 1.0f0 / 4.0f0
    grid = [0.0f0, dx, 2.0f0*dx, 3.0f0*dx, 1.0f0]
    ϕ(θ, x) = begin
        reshape(θ, 1, :) * acFun.((x .- grid) / dx)
    end
    θ = zeros(Float32, 5) #for example
    (ϕ, θ)
end

function get_loss(ϕ)
    loss(θ, p) = begin
        u1, u2, u3, u4, u5 = θ
        (1/48)*(16 - 18*u2 + 392*u2^2 - 12*u3 -
                380*u2*u3 + 392*u3^2 - 6*u4 -
                380*u3*u4 + 392*u4^2 + 95*u5 -
                380*u4*u5 + 244*u5^2)
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
