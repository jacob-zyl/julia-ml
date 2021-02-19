using Flux
using Zygote
using ForwardDiff
using ReverseDiff


###
###
###
###

# func(x, w) = (w * x)
func(x, w) = σ.(w*x)
# func(x, w) = sin.(w*x)

w = rand(Float32, 1, 2)

x = [ones(Float32, 3) 3ones(Float32, 3)]'

x0 = [ones(Float32, 3) zeros(Float32, 3)]'

dfunc_pf(x, w) = begin pushforward(ξ -> func(ξ, w), x)(x0) end
dfunc_pb(x, w) = begin pullback(ξ -> func(ξ, w), x)[2](ones(Float32, 1, 3))[1] end
dfunc_fd(x, w) = ForwardDiff.jacobian(ξ -> func(ξ, w), x)

loss_pf(w) = sum(dfunc_pf(x, w))
loss_fd(w) = sum(dfunc_fd(x, w))
loss_pb(w) = sum(dfunc_pb(x, w))

ForwardDiff.gradient(loss_pb, w)
ReverseDiff.gradient(loss_pf, w)
Zygote.gradient(loss_pb, w)

###
###
###
###

model = Dense(3, 1, σ)
θ, re = Flux.destructure(model)

ϕ(x, θ) = re(θ)(x)

x = [ones(Float32, 1, 5); 3f0ones(Float32, 1, 5); 4.0f0ones(Float32, 1, 5)]

A = ForwardDiff.jacobian(t -> ϕ(x, t), θ)

using LinearAlgebra
F = svd(A)

###
###
###
###


using GalacticOptim, Optim
rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0,100.0]

prob = OptimizationProblem(rosenbrock,x0,p)
sol = solve(prob,NelderMead())

rb_strange(x, p) = begin
    a, b = x
    (p[1] - a)^2 + p[2] * (b - a^2)^2
end
x0_strange = [0.0 0.0]
prob_strange = OptimizationProblem(rb_strange, x0_strange, p)
sol_strange = solve(prob_strange, NelderMead())
