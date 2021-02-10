#=
# Comes from pinn2.jl
=#
using LinearAlgebra
using Flux
using Zygote
using ForwardDiff
using GalacticOptim, Optim

using Plots
const DIM = 1
f = Chain(Dense(DIM, 50, tanh), Dense(50, 1))
θ = params(f)

df = x -> Zygote.pullback(f, x)[2](ones(1, size(x, 2)))[1]

x̂ = range(0, stop=2π, length=100) |> collect |> a -> reshape(a, 1, :)
x_test = 0:0.2:2π |> collect |> a -> reshape(a, 1, :)

function result(x_test)
    scatter(x_test', f(x_test)')
    plot!(x_test', sin.(x_test)')
    annotate!(3, 0, norm(f(x_test) - sin.(x_test)))
end

eq_res = (x) -> (norm(df(x) - cos.(x)))

loss_original(x) = begin
    r = eq_res(x) + (norm(f([0f0])))
end

loss_hard(x) = begin
    r = loss_original(x)
    r + log1p(r / 1f-3)
end

loss_harder(x) = begin
    r = loss_hard(x)
    l = loss_original(x)
    r + l*l
end


data = Flux.Data.DataLoader((x̂,), batchsize = 10, shuffle=true)
Flux.Optimise.@epochs 2000 Flux.Optimise.train!(
    loss_original,
    θ,
    data,
    ADAM(),
);
result(x_test)
