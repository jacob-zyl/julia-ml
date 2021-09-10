using LinearAlgebra
using Statistics: mean
using Flux
using Zygote
using ForwardDiff
using GalacticOptim, Optim

using Plots

macro D(f)
    return :((x, θ) -> Zygote.pullback(y -> $f(y, θ), x)[2](ones(1, size(x, 2)))[1] )
end

function D(f)
    return (x, θ) -> Zygote.pullback(y -> f(y, θ), x)[2](ones(1, size(x, 2)))[1]
end

const DIM = 1
const BATCH_SIZE = 100

f = Chain(Dense(DIM, 20, tanh), Dense(20, 1))
θ, re = Flux.destructure(f)
ϕ(x, θ) = re(θ)(x)

x̂ = range(0, stop=2π, length=100) |> collect |> a -> reshape(a, 1, :)
x_test = 0:0.2:2π |> collect |> a -> reshape(a, 1, :)

function get_loss(f)
    x_bd_1 = [0f0]
    x_pde = x̂
    #fx(y, θ) = Zygote.pullback((x -> f(x, θ)), y)[2](ones(1, size(y, 2)))[1]
    # This works but I want it faster
    #ϕx(y, θ) = Zygote.pullback((x -> ϕ(x, θ)), y)[2](ones(1, BATCH_SIZE))[1]
    # UPDATE: the previous two ways are identical, a ten-time comparasion
    #         experiment shows that:
    #  |w/o const | 5 5 3 3 3 4 2 3 3 4|
    #  |w/  const | 7 3 4 3 3 3 2 2 6 4|
    #fx = @D(f)
    fx = D(f)
    loss(θ, p) = begin
        eq_res = fx(x_pde, θ) - cos.(x_pde)
        eq_residual = mean(abs2, eq_res)
        bd_residual = mean(abs2, f(x_bd_1, θ))
        bd_residual + eq_residual
    end
    loss_hard(θ, p) = begin
        r = loss(θ, p)
        r + log1p(1f5*r)*1f-3
    end
end

opt_f = OptimizationFunction(get_loss(ϕ), GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(opt_f, θ)
sol = solve(prob, Optim.BFGS())

function result(sol=sol, x_test=x_test)
    y_test = ϕ(x_test, sol.minimizer)
    y_ref = sin.(x_test)
    p = plot(x_test', y_test')
    plot!(p, x_test', y_ref')
    annotate!(p, 3, 0, norm(y_test - y_ref))
    p
end

p = result()
