using LinearAlgebra
using Flux
using Zygote
using ForwardDiff
using GalacticOptim, Optim

using Plots
const DIM = 1
f = Chain(Dense(DIM, 50, tanh), Dense(50, 1))
# f = x -> [1f0 2f0] * x
# f = x -> sum(dot.(x, [2f0 1f0; 0f0 3f0] * x), dims=1)
θ = params(f)

bv(i, n) = [j == i for j = 1:n]
# The differential operators
#
grad(f) = x -> ForwardDiff.gradient(f, x) # Zygote.gradient fails for closure

sum1(f) = x -> sum(f(x))
grad1(f) = grad(sum1(f))
split(f, n) = [x -> bv(i, n)' * f(x) for i = 1:n]
split(f) = split(f, DIM)

# Generated total derivative by operators
#
df = grad1(f)
ddfx, = split(df)
dddfxx, = split(grad1(df))

x̂ = range(0, stop=2π, length=100) |> a -> reshape(a, 1, :)
x_test = 0:0.2:2π |> a -> reshape(a, 1, :)
ŷ = sin.(x̂)

function result(x_test)
    plot(x_test', f(x_test)')
    plot!(x_test', sin.(x_test)')
    annotate!(3, 0, norm(f(x_test) - sin.(x_test)))
end

r(x) = ddfx(x)

eq_res = (x) -> sqrt(norm(r(x)))
loss_hard(x, y) = begin
    r = sqrt(norm(y - f(x)))
    r + log1p(r / 1f-3)
end

loss_original(x, y) = begin
    r = sqrt(norm(y - f(x)))
end

loss_harder(x, y) = begin
    r = sqrt(norm(y - f(x)))
    r + log1p(r / 1f-2) + r*r
end


# opt_f = OptimizationFunction(loss_hard, GalacticOptim.AutoZygote())
# prob = OptimizationProblem(opt_f, θ)
# sol = solve(prob, Optim.BFGS())
data = Flux.Data.DataLoader((x̂, ŷ), batchsize = 10, shuffle=true)
Flux.Optimise.@epochs 5000 Flux.Optimise.train!(
    loss_harder,
    θ,
    data,
    ADAM(),
);
result(x_test)
