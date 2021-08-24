###
### This code fails.
###
###
using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf, GLMakie

using Zygote: dropgrad, Buffer, jacobian
using ForwardDiff: derivative

using JLD

const nu = 0.03
const γ = 1.4
const NK = 4

show_result(filename) = begin
    time = load(filename, "time")
    mesh = load(filename, "mesh")
    data = load(filename, "data")
    lines(mesh, data[1, :])
end


## A Short Explain on Data Structure
#
#  Physical information is organized by two parts: the mesh and the data. The
#  mesh is just an vector since there is no complex structure in 1D, but usually
#  there should be two extra matrice: the first matrix records nodes of each
#  element and the second records boundary conditions. The data records
#  computational parameters on each node. Here in this program, the data is a
#  6×#Nodes matrix with each column record computational parameters on the
#  corresponding node. The first two elements of each column of matrix data are
#  conservative variable 𝑤₁ and its spacial derivative ∂𝑤₁/∂x, the third and
#  fourth elements correspond to 𝑤₂, and the fifth and sixth to 𝑤₃

train(N, dt, T) = begin
    mesh = get_mesh(N)
    data = get_data(N)
    loss_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())

    time = 0.0
    iters = 0
    while time < T
        prob = OptimizationProblem(loss_f, data, (dt, mesh, data))
        sol = solve(prob, ConjugateGradient())
        data = sol.minimizer
        @printf "%e\n" sol.minimum
        time += dt
        iters += 1
        @save "sod"*(@sprintf "%04i" iters)*".jld" data mesh time
    end
end

loss(data, fem_params) = begin
    dt, mesh, data_init = fem_params
    ng = length(mesh)
    ne = ng - 1

    # buf = Buffer(data)
    # buf[:, :] = data[:, :]
    # buf[1, 1] = dropgrad(data[1, 1])
    # buf[1, end] = dropgrad(data[1, end])
    # buf[3, 1] = dropgrad(data[3, 1])
    # buf[3, end] = dropgrad(data[3, end])
    # buf[5, 1] = dropgrad(data[5, 1])
    # buf[5, end] = dropgrad(data[5, end])
    # data = copy(buf)

    loss = 0
    for iters in 1:ne
        indice = [iters, iters+1]
        elnode = @views mesh[indice]
        eldata = @views data[:, indice]
        elinit = @views data_init[:, indice]
        loss += element_loss(elnode, eldata, elinit, dt)
    end
    loss
end

element_loss(nodes, data, init, dt) = begin

    # data = [𝑤₁; 𝑤₂; 𝑤₃]
    rdt = 1.0 / dt

    # # Long live the isoparametric elements!
    # fcoord = [nodes'; ones(1, 2)]
    # coord = Hi * vec(fcoord)

    # transform the data into local coordinate
    Δ = nodes[2] - nodes[1]
    ratio = [1.0, 0.5Δ, 1.0, 0.5Δ, 1.0, 0.5Δ]
    f = data .* ratio
    finit = init .* ratio

    w1data = f[1:2, :]
    w2data = f[3:4, :]
    w3data = f[5:6, :]

    w1initdata = finit[1:2, :]
    w2initdata = finit[3:4, :]
    w3initdata = finit[5:6, :]

    w1 = Hi * vec(w1data)
    w1init = Hi * vec(w1initdata)
    w2 = Hi * vec(w2data)
    w2init = Hi * vec(w2initdata)
    w3 = Hi * vec(w3data)
    w3init = Hi * vec(w3initdata)

    wx1 = Hxi * vec(w1data)
    wx1init = Hxi * vec(w1initdata)
    wx2 = Hxi * vec(w2data)
    wx2init = Hxi * vec(w2initdata)
    wx3 = Hxi * vec(w3data)
    wx3init = Hxi * vec(w3initdata)

    # wxx1 = Hxxi * vec(w1data)
    # wxx1init = Hxxi * vec(w1initdata)
    # wxx2 = Hxxi * vec(w2data)
    # wxx2init = Hxxi * vec(w2initdata)
    # wxx3 = Hxxi * vec(w3data)
    # wxx3init = Hxxi * vec(w3initdata)

    res1 = @. (w1 - w1init) * rdt + wx2
    res2 = @. (w2 - w2init) * rdt + (0.5 * (γ - 3.0) * w2^2 / w1^2 * wx1 +
        (3.0 - γ) * w2 / w1 * wx2 + (γ - 1.0) * wx3)
    res3 = @. (w3 - w3init) * rdt + (-γ * w2 * w3 / w1^2 * wx1 +
        (γ * w3 / w1 - 1.5 * (γ - 1.0) * w2^2 / w1^2) * wx2 + γ * w2 / w1 * wx3)
    0.5Δ * (W ⋅ (res1.^2 + res2.^2 + res3.^2))
end

using FastGaussQuadrature, Roots
const P, W = gausslegendre(NK)

# Since the code here is settled down, and it works well, so I won't make
# further enhancement. However, this very mathematical way of defining the basis
# function is clumsy in programming. A better way is to define functions H1(x),
# H2(x), H3(x), H4(x). In one dimention cases, the four functions framework is
# better than the two functions framework, but in higher dimensional cases, it
# is far better.
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

get_mesh(N) = begin
    range(0, stop=5, length=N+1) |> collect
end

get_primary_data(mesh) = begin
    ρ = map(mesh) do x
        if x < 2.5
            return 1.0
        else
            return 0.125
        end
    end
    u = map(mesh) do x
        return 0.0
    end
    p = map(mesh) do x
        if x < 2.5
            return 1.0
        else
            return 0.1
        end
    end
    (ρ, u, p)
end

get_primary_data(w1, w2, w3) = begin
    ρ = w1
    u = @. w2 / w1
    p = @. (γ - 1.0) * (w3 - 0.5 * w2^2 / w1)
    (ρ, u, p)
end


get_conservative_data(ρ, u, p) = begin
    w1 = ρ
    w2 = @. ρ * u
    w3 = @. 0.5 * ρ * u^2 + p / (γ - 1.0)
    (w1, w2, w3)
end

get_conservative_data(mesh) = begin
    ρ, u, p = get_primary_data(mesh)
    get_conservative_data(ρ, u, p)
end

get_data(N) = begin
    mesh = get_mesh(N)
    data_tuple = get_conservative_data(mesh)
    data = [data_tuple[1] zero(mesh) data_tuple[2] zero(mesh) data_tuple[3] zero(mesh)]' |> collect
end
