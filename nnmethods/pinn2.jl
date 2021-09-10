#=
# It is very frustrating that this version doesn't work
#
#
# Sun Feb  7 22:00:39 2021 UPDATE::::
# It finally works!!!
#
#
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

# bv(i, n) = [j == i for j = 1:n]
# vecgroup = [bv(i, DIM) for i = 1:DIM]

# # The differential operators
# #
# grad(f) = x -> ForwardDiff.gradient(f, x) # Zygote.gradient fails for closure

# sum1(f) = x -> sum(f(x))
# grad1(f) = grad(sum1(f))

# sum2(g) = x -> sum(g(x), dims = 2)
# part(g) = [x -> vecgroup[i] ⋅ g(x) for i = 1:DIM]
# grad2(f) = grad.(part(sum2(f)))

# Generated total derivative by operators
#
# df = grad1(f)
# ddfx, = grad2(df)
# dddfxx, = grad2(ddfx)
#
# df = y -> Zygote.gradient(y) do x
#     Zygote.forwarddiff(sum ∘ f, x)
#     end[1]

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
Flux.Optimise.@epochs 1000 Flux.Optimise.train!(
    loss_original,
    θ,
    data,
    ADAM(),
);
#result(x_test)
