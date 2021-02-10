using Flux
using Zygote

mytranslation(point::Array{AbstractFloat, 1}) = begin
    point + [1.0; 1.0]
end

# f = Chain(Dense(2, 5, tanh), Dense(5, 1))
# x_test = [1f0 2f0; 3f0 4f0]
# gradf = x -> ForwardDiff.gradient(sum ∘ f, x)

# df = x_test -> Zygote.pullback(f, x_test)[2](ones(1, size(x_test, 2)))[1]

f = Chain(Dense(1, 5, tanh), Dense(5, 1))
x_test = [1f0 2f0]

gradf = x -> ForwardDiff.gradient(sum ∘ f, x)

df = x -> Zygote.pullback(f, x)[2](ones(1, size(x, 2)))[1]

macro D(f)
    return :((x, θ) -> Zygote.pullback(y -> $f(y, θ), x)[2](ones(1, size(x, 2)))[1] )
end
