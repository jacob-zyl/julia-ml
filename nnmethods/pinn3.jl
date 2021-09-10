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

function mysum(f::Array{T, 2}) where T <: Real
    f * ones(size(f))'
end

function result(sol, x_test)
    y_test = ϕ(x_test, sol.minimizer)
    y_ref = sin.(x_test)
    plot(x_test', y_test')
    plot!(x_test', y_ref')
    annotate!(3, 0, norm(y_test - y_ref))
end

function get_loss(ϕ)
    # The differential operators
    #
    grad(f) = x -> ForwardDiff.gradient(sum ∘ f, x)
    mygrad(f) = x -> ForwardDiff.jacobian(mysum ∘ f, x)
    grad2(phi) = (x, t) -> (mygrad(y -> phi(y, t)))(x)
    # Zygote.gradient fails for closure

    bv(i, n) = [j == i for j = 1:n]
    split(f, n) = [x -> bv(i, n)' * f(x) for i = 1:n]
    split(f) = split(f, DIM)

    split2(phi, n) =[(x, t) -> bv(i, n)' * phi(x, t) for i = 1:n]
    split2(phi) = split2(phi, DIM)

    sgrad = split ∘ grad
    sgrad2 = split2 ∘ grad2
    # Generated total derivative by operators
    #
    fx, = sgrad(f)
    fxx, = sgrad(fx)

    # ϕx, = sgrad2(ϕ)
    ϕx = grad2(ϕ)
    ϕxx, = sgrad2(ϕx)

    # loss_hard(θ, p) = begin
    #     r = loss(θ, p)
    #     r + log1p(r / 1f-3)
    # end

    # loss_harder(θ, p) = begin
    #     r = loss_hard(θ, p)
    #     l = loss(θ, p)
    #     r + l*l
    # end

    loss(θ, p) = begin
        x_bd_Dirichlet = [0f0]
        x_pde = x̂
        eq_res = ϕx(x_pde, θ) - cos.(x_pde)
        eq_residual = mean(abs2, eq_res)
        bd_residual = mean(abs2, ϕ(x_bd_Dirichlet, θ))
        bd_residual + eq_residual
    end
end

opt_f = OptimizationFunction(get_loss(ϕ), GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(opt_f, θ)
sol = solve(prob, Optim.BFGS())
#result(sol, x_test)
