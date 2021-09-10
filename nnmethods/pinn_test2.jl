using LinearAlgebra
using Flux
using Zygote
using ForwardDiff

# ϕ = Chain(Dense(2, 50, tanh),
#           Dense(50, 1))
# ϕ = x -> [1f0 2f0] * x
f = x -> sum(dot.(x, [2f0 1f0; 0f0 3f0] * x), dims=1)

part(f) = (x -> [1f0; 0f0] ⋅ f(x), x -> [0f0; 1f0] ⋅ f(x))
sum2(f) = x -> sum(f(x), dims=2)
sum1(f) = x -> sum(f(x))
grad(f) = x -> ForwardDiff.gradient(f, x) # Zygote.gradient fails for closure
grad1 = grad ∘ sum1
grad2(f) = grad.(part(sum2(f)))

df = grad1(f)
ddfx, ddfy = grad2(df)
dddfxx, dddfxy = grad2(ddfx)
dddfyx, dddfyy = grad2(ddfy)

# variables for testing
x = [1f0; 2f0];
x1 = reshape(x, 2, 1);
x2 = [x x];
x3 = [x x x];

print("dddfxx(x) = \n", dddfxx(x));
print("dddfxy(x1) = \n", dddfxy(x1), "\n")
print("dddfyx(x2) = \n", dddfxy(x2), "\n")
print("dddfyy(x3) = \n", dddfxy(x3), "\n")
