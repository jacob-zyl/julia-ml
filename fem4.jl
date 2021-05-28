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
    NUM = 10
    dx = 1.0f0 / NUM
    grid = range(0.0f0, 1.0f0, length=NUM+1) |> collect
    ϕ(θ, x) = begin
        reshape(θ, 1, :) * acFun.(NUM * (x .- grid))
    end
    θ = zeros(Float32, NUM+1) #for example
    (ϕ, θ)
end

function get_loss(ϕ)
    loss(θ, p) = begin
        u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11 = θ
        (1/300)*(100 - 6*u10 + 6020*u10^2 +
                 599*u11 - 5990*u10*u11 + 3310*u11^2 -
                 54*u2 + 6020*u2^2 - 48*u3 -
                 5990*u2*u3 + 6020*u3^2 - 42*u4 -
                 5990*u3*u4 + 6020*u4^2 - 36*u5 -
                 5990*u4*u5 + 6020*u5^2 - 30*u6 -
                 5990*u5*u6 + 6020*u6^2 - 24*u7 -
                 5990*u6*u7 + 6020*u7^2 - 18*u8 -
                 5990*u7*u8 + 6020*u8^2 - 12*u9 -
                 5990*u10*u9 - 5990*u8*u9 + 6020*u9^2)
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
