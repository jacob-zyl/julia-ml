using LinearAlgebra, Statistics, StaticArrays
using Flux, DiffEqFlux
using GalacticOptim, Optim
using Plots

# A = randn(Float32, 1000, 1000)
# x = randn(Float32, 1000)

# b = A * x

A = [4.0f0 2.0f0; 2.0f0 1.0f0]

b = [0.0f0; 1.0f0]

x = randn(Float32, 2)

loss(theta, p) = begin
    mean(abs, A * theta - b)
end

theta = 2.0f0ones(Float32, size(x))
opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
prob = OptimizationProblem(opt_f, theta)
#sol = solve(prob, ADAM(); maxiters=100)
