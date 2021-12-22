using LinearAlgebra, Statistics
using GalacticOptim, Optim
using Printf, CairoMakie
using CSV, DataFrames

using Zygote: dropgrad, Buffer, jacobian
using ForwardDiff 

using JLD

using Dierckx

const γ = 1.4
const NK = 4

const ρl = 1.0
const pl = 1.0
const ρr = 0.125
const pr = 0.1

show_result(filename) = begin
    exact_data = CSV.File(
        "exact_sod_output", delim="   ", header=0, skipto=3,
        select=["Column1", "Column2", "Column3", "Column4", "Column5"]) |> DataFrame
    x        = map(t -> parse(Float64, t), exact_data.Column1)
    density  = map(t -> parse(Float64, t), exact_data.Column2)
    pressure = map(t -> parse(Float64, t), exact_data.Column3)
    velocity = map(t -> parse(Float64, t), exact_data.Column4)
    energy   = map(t -> parse(Float64, t), exact_data.Column5)

    time = load(filename, "time")
    mesh = load(filename, "mesh")
    data = load(filename, "data")
    w1 = data[1, :]
    w2 = data[3, :]
    w3 = data[5, :]
    ρ, u, p = get_primary_data(w1, w2, w3)
    ϵ = @. p / ( ρ * (γ - 1.0) )
    fig = Figure()
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[1, 2])
    ax3 = Axis(fig[2, 1])
    ax4 = Axis(fig[2, 2])
    scatter!(ax1, mesh, u, markersize=5, label="Numerical") # velocity
    scatter!(ax2, mesh, ρ, markersize=5, label="Numerical") # density
    scatter!(ax3, mesh, p, markersize=5, label="Numerical") # pressure
    scatter!(ax4, mesh, ϵ, markersize=5, label="Numerical") # total energy
    lines!(ax1, x, velocity, color=:red, label="Analytical")
    lines!(ax2, x, density, color=:red, label="Analytical")
    lines!(ax3, x, pressure, color=:red, label="Analytical")
    lines!(ax4, x, energy, color=:red, label="Analytical")
    ax1.ylabel="Velocity"
    ax2.ylabel="Density"
    ax3.ylabel="Pressure"
    ax4.ylabel="Energy"
    Legend(fig[:, 3], ax1)
    fig
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
		spline!(mesh, data)
        @printf "%e\n" sol.minimum
        time += dt
        iters += 1
        @save "sod6/"*(@sprintf "%04i" iters)*".jld" data mesh time
    end
end

loss(data, fem_params) = begin
    dt, mesh, data_init = fem_params
    ng = length(mesh)
    ne = ng - 1

    buf = Buffer(data)
    buf[:, :] = data[:, :]

	buf[2, :] = dropgrad(data[2, :])
	buf[4, :] = dropgrad(data[4, :])
	buf[6, :] = dropgrad(data[6, :])

    buf[1, 1] = dropgrad(data[1, 1])
    buf[1, end] = dropgrad(data[1, end])
    buf[3, 1] = dropgrad(data[3, 1])
    buf[3, end] = dropgrad(data[3, end])
    buf[5, 1] = dropgrad(data[5, 1])
    buf[5, end] = dropgrad(data[5, end])

    data = copy(buf)

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
    det = 0.5Δ
    ratio = [1.0, 0.5Δ, 1.0, 0.5Δ, 1.0, 0.5Δ]
    f = data .* ratio
    finit = init .* ratio

    w1data = @view f[1:2, :]
    w2data = @view f[3:4, :]
    w3data = @view f[5:6, :]

    w1initdata = @view finit[1:2, :]
    w2initdata = @view finit[3:4, :]
    w3initdata = @view finit[5:6, :]

    pdata = @. (γ - 1.0) * (w3data[1, :] - 0.5 * w2data[1, :]^2 / w1data[1, :])

    w1     = quad_on_element(w1data, det)
    w1init = quad_on_element(w1initdata, det)
    w2     = quad_on_element(w2data, det)
    w2init = quad_on_element(w2initdata, det)
    w3     = quad_on_element(w3data, det)
    w3init = quad_on_element(w3initdata, det)

    flux1_left  = @views w2data[1, 1]
    flux1_right = @views w2data[1, 2]

    flux2_left  = @views w2data[1, 1]^2 / w1data[1, 1] + pdata[1]
    flux2_right = @views w2data[1, 2]^2 / w1data[1, 2] + pdata[2]

    flux3_left  = @views (w2data[1, 1] / w1data[1, 1]) * (w3data[1, 1] + pdata[1])
    flux3_right = @views (w2data[1, 2] / w1data[1, 2]) * (w3data[1, 2] + pdata[2])

    res1 = (w1 - w1init) * rdt + flux1_right - flux1_left
    res2 = (w2 - w2init) * rdt + flux2_right - flux2_left
    res3 = (w3 - w3init) * rdt + flux3_right - flux3_left
    (res1.^2 + res2.^2 + res3.^2)
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

Hx1(x; ξ) = ForwardDiff.derivative(p -> H1(p; ξ=ξ), x)
Hx2(x; ξ) = ForwardDiff.derivative(p -> H2(p; ξ=ξ), x)

Hxx1(x; ξ) = ForwardDiff.derivative(p -> Hx1(p; ξ=ξ), x)
Hxx2(x; ξ) = ForwardDiff.derivative(p -> Hx2(p; ξ=ξ), x)

H(x::Real) = [H1(x; ξ=-1) H2(x; ξ=-1) H1(x; ξ=1) H2(x; ξ=1)]
H(x::Vector) = vcat(H.(x)...)
Hx(x::Real) = [Hx1(x; ξ=-1) Hx2(x; ξ=-1) Hx1(x; ξ=1) Hx2(x; ξ=1)]
Hx(x::Vector) = vcat(Hx.(x)...)
Hxx(x::Real) = [Hxx1(x; ξ=-1) Hxx2(x; ξ=-1) Hxx1(x; ξ=1) Hxx2(x; ξ=1)]
Hxx(x::Vector) = vcat(Hxx.(x)...)

const Hi = H(P) # matrix NK × 4
const Hxi = Hx(P)
const Hxxi = Hxx(P)

const WHi = Hi' * W

value_on_gaussian_points(data) = begin
    Hi * vec(data)
end

derivative_on_gaussian_points(data) = begin
    Hxi * vec(data)
end

quad_on_element(data, det) = begin
    value = value_on_gaussian_points(data)
    det * (W ⋅ value)
    # det * (WHi ⋅ vec(data))  # a faster implementation
end

get_mesh(N) = begin
    range(0.0, stop=1.0, length=N+1) |> collect
end

get_primary_data(mesh) = begin
    ρ = map(mesh) do x
        if x < 0.5 * (mesh[end] - mesh[1])
            return ρl
        else
            return ρr
        end
    end
    u = map(mesh) do x
        return 0.0
    end
    p = map(mesh) do x
        if x < 0.5 * (mesh[end] - mesh[1])
            return pl
        else
            return pr
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

spline!(mesh, data) = begin
	itp1 = Spline1D(mesh, data[1, :])
	itp2 = Spline1D(mesh, data[3, :])
	itp3 = Spline1D(mesh, data[5, :])
	data[2, :] = Dierckx.derivative(itp1, mesh)
	data[4, :] = Dierckx.derivative(itp2, mesh)
	data[6, :] = Dierckx.derivative(itp3, mesh)
end
