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
    NUM = 20
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
        u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21 = θ
        (1/1200)*(400 - 66*u10 + 48040*u10^2 -
                  60*u11 - 47980*u10*u11 +
                  48040*u11^2 - 54*u12 -
                  47980*u11*u12 + 48040*u12^2 -
                  48*u13 - 47980*u12*u13 +
                  48040*u13^2 - 42*u14 -
                  47980*u13*u14 + 48040*u14^2 -
                  36*u15 - 47980*u14*u15 +
                  48040*u15^2 - 30*u16 -
                  47980*u15*u16 + 48040*u16^2 -
                  24*u17 - 47980*u16*u17 +
                  48040*u17^2 - 18*u18 -
                  47980*u17*u18 + 48040*u18^2 -
                  12*u19 - 47980*u18*u19 +
                  48040*u19^2 - 114*u2 + 48040*u2^2 -
                  6*u20 - 47980*u19*u20 + 48040*u20^2 +
                  2399*u21 - 47980*u20*u21 +
                  25220*u21^2 - 108*u3 - 47980*u2*u3 +
                  48040*u3^2 - 102*u4 - 47980*u3*u4 +
                  48040*u4^2 - 96*u5 - 47980*u4*u5 +
                  48040*u5^2 - 90*u6 - 47980*u5*u6 +
                  48040*u6^2 - 84*u7 - 47980*u6*u7 +
                  48040*u7^2 - 78*u8 - 47980*u7*u8 +
                  48040*u8^2 - 72*u9 - 47980*u10*u9 -
                  47980*u8*u9 + 48040*u9^2)
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
