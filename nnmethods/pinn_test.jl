using LinearAlgebra
using Flux
using Zygote
using ForwardDiff

# ϕ = Chain(Dense(2, 50, tanh),
#           Dense(50, 1))
# ϕ = x -> [1f0 2f0] * x
ϕ = x -> dot.(x, [2f0 1f0; 0f0 3f0] * x)

# ϕ₁ = x -> ϕ(x)[1]
# ϕ₂ = x -> ϕ(x)[2]

f = x -> sum(ϕ(x))


dϕ1(x::Array{T,2}) where T <: Real = begin
    ndims = size(x)[1]
    npoints = size(x)[2]
    A = zeros(ndims, npoints)
    for i in 1:npoints
        A[:, i] = ForwardDiff.jacobian(ϕ, x[:, i])
    end
    A
end

dϕ1(x::Array{T,1}) where T <: Real = begin
    A = ForwardDiff.jacobian(ϕ, x)
    reshape(A, size(x)[1], 1)
    end

dϕ(x) = begin
    if isa(x, Array{<:Real, 1})
        A = ForwardDiff.jacobian(ϕ, x)
        A = reshape(A, size(x)[1], 1)
    elseif isa(x, Array{<:Real, 2})
        ndims = size(x)[1]
        npoints = size(x)[2]
        A = zeros(ndims, npoints)
        for i in 1:npoints
            A[:, i] = ForwardDiff.jacobian(ϕ, x[:, i])
        end
    else
        throw(DomainError(x))
    end
    A
    end

df = x -> ForwardDiff.gradient(f, x)
ddf = x -> ForwardDiff.jacobian(df, x)
dhf = x -> ForwardDiff.hessian(f, x)

pdf = x -> Zygote.gradient(f, x)[1]
pddf = x -> Zygote.hessian(f, x)

dpddf = x -> ForwardDiff.jacobian(pddf, x)

dfx = x -> [1f0; 0f0] ⋅ sum(df(x), dims=2)
dfy = x -> [0f0; 1f0] ⋅ sum(df(x), dims=2)

ddfx = x -> ForwardDiff.gradient(dfx, x) # This is [∂ₓ∂ₓf; ∂ₜ∂ₓf]
ddfy = y -> ForwardDiff.gradient(dfy, y) # This is [∂ₓ∂ₜf; ∂ₜ∂ₜf]
