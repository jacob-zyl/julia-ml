# So strange this doesn't work
#
# Sun Feb  7 23:54:00 2021 UPDATE
# Finally works in a interesting way where Zygote still not possible as ADtype of Optimization function.
using LinearAlgebra
using Statistics: mean
using Flux
using Zygote
using ForwardDiff
using GalacticOptim, Optim

using Plots
const DIM = 1
f = Chain(Dense(DIM, 50, tanh), Dense(50, 1))
θ, re = Flux.destructure(f)
ϕ(x, θ) = re(θ)(x)

x̂ = range(0, stop=2π, length=100) |> collect |> a -> reshape(a, 1, :)
x_test = 0:0.2:2π |> collect |> a -> reshape(a, 1, :)

function result(sol, x_test)
    y_test = ϕ(x_test, sol.minimizer)
    y_ref = sin.(x_test)
    p = plot(x_test', y_test')
    plot!(p, x_test', y_ref')
    annotate!(p, 3, 0, norm(y_test - y_ref))
    p
end

function get_loss(f)
    x_bd_1 = [0f0]
    x_pde = x̂
    θ, re = Flux.destructure(f)
    ϕ(x, θ) = re(θ)(x)
    ϕx(θ) = Zygote.pullback((x -> ϕ(x, θ)), x_pde)[2](ones(1, size(x_pde, 2)))[1]
    loss(θ, p) = begin
        eq_res = ϕx(θ) - cos.(x_pde)
        eq_residual = mean(abs2, eq_res)
        bd_residual = mean(abs2, ϕ(x_bd_1, θ))
        bd_residual + eq_residual
    end
    loss_hard(θ, p) = begin
        r = loss(θ, p)
        r + log1p(r / 1f-3)
    end

    loss_harder(θ, p) = begin
        r = loss_hard(θ, p)
        l = loss(θ, p)
        r + l*l
    end

end

opt_f = OptimizationFunction(get_loss(f), GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(opt_f, θ)
sol = solve(prob, Optim.BFGS())
p = result(sol, x_test)
