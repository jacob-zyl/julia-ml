using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf, CairoMakie

using Zygote: dropgrad, Buffer, jacobian
using ForwardDiff: derivative

using JLD

const nu = 0.05
const NK = 4

train(N, dt, T) = begin
    mesh = get_mesh(N)
    data = get_data(N)
    loss_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())

    time = 0.0
    iters = 0
    @save "burgers"*(@sprintf "%04i" iters)*".jld" data mesh time
    while time < T
        prob = OptimizationProblem(loss_f, data, (dt, mesh, data))
        sol = solve(prob, ConjugateGradient())
        data = sol.minimizer
        @printf "%e\n" sol.minimum
        time += dt
        iters += 1
        @save "burgers"*(@sprintf "%04i" iters)*".jld" data mesh time
    end
end

get_mesh(N) = begin
    sinpi.((0:N) ./ 2N) |> collect
end

get_data(N) = begin
    data = zeros(2, N+1)
    data[1, :] = 1.0 .- get_mesh(N)
    data[2, :] .= -1
    data
end

loss(data, fem_params) = begin
    dt, mesh, data_init = fem_params
    ng = length(mesh)
    ne = ng - 1

    buf = Buffer(data)
    buf[:, :] = data[:, :]
    buf[1, 1] = dropgrad(data[1, 1])
    buf[1, end] = dropgrad(data[1, end])
    data = copy(buf)

    sum = 0
    for iters in 1:ne
        indice = [iters, iters+1]
        elnode = @views mesh[indice]
        eldata = @views data[:, indice]
        elinit = @views data_init[:, indice]
        sum += element_loss(elnode, eldata, elinit, dt)
    end
    sum
end

element_loss(nodes, data, init, dt) = begin
    Δ = nodes[2] - nodes[1]
    ratio = [1.0, 0.5Δ]

    # # Long live the isoparametric elements!
    # fcoord = [nodes'; ones(1, 2)]
    # coord = Hi * vec(fcoord)

    f = data .* ratio
    finit = init .* ratio
    rdt = 1.0 / dt
    u = Hi * vec(f)
    uinit = Hi * vec(finit)

    ux = Hxi * vec(f) * 2Δ^-1
    uxx = Hxxi * vec(f) * 4Δ^-2

    residual = @. rdt * (u - uinit) + u * ux - nu * uxx

    (W ⋅ residual.^2)
end

using FastGaussQuadrature, Roots
const P, W = gausslegendre(NK)

# Since the code here is settled down, and it works well, so I won't make
# further enhancement. However, this very mathematical way of defining the
# basis function is clumsy in programming. A better way is to define functions
# H1(x), H2(x), H3(x), H4(x). In one dimention cases, the four functions
# framework is better than the two functions framework, but in higher
# dimensional cases, it is far better.
H1(x; ξ) =  0.25*(1.0 + ξ*x)^2*(2.0 - ξ*x)
H2(x; ξ) =  0.25*(1.0 + ξ*x)^2*(x - ξ)

Hx1(x; ξ) = derivative(p -> H1(p; ξ=ξ), x)
Hx2(x; ξ) = derivative(p -> H2(p; ξ=ξ), x)

Hxx1(x; ξ) = derivative(p -> Hx1(p; ξ=ξ), x)
Hxx2(x; ξ) = derivative(p -> Hx2(p; ξ=ξ), x)

H(x::Real) = [H1(x; ξ=-1) H2(x; ξ=-1) H1(x; ξ=1) H2(x; ξ=1)]
H(x::Vector) = vcat(H.(x)...)
Hx(x::Real) = [Hx1(x; ξ=-1) Hx2(x; ξ=-1) Hx1(x; ξ=1) Hx2(x; ξ=1)]
Hx(x::Vector) = vcat(Hx.(x)...)
Hxx(x::Real) = [Hxx1(x; ξ=-1) Hxx2(x; ξ=-1) Hxx1(x; ξ=1) Hxx2(x; ξ=1)]
Hxx(x::Vector) = vcat(Hxx.(x)...)

const Hi = H(P)
const Hxi = Hx(P)
const Hxxi = Hxx(P)

function f_exact(x)
    ū * (2/(1 + exp(ū*(x-1)/nu)) - 1)
end

function fx_exact(x)
    derivative(f_exact, x)
end

u_im(x) = (x - 1.0)/(x + 1.0) - exp(-x/nu)
const ū = find_zero(u_im, 1)

