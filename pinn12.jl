using LinearAlgebra
using Statistics: mean
using Flux
using DiffEqFlux
using Zygote
using GalacticOptim
using Optim
using Quadrature
using Printf
using Plots
plotlyjs()
theme(:vibrant)

const DIM = 2
const HIDDEN = 20
const BATCH_SIZE = 100
const P_SIZE = DIM * HIDDEN + HIDDEN + HIDDEN * 1 + 1
const P1 = DIM * HIDDEN
const P2 = HIDDEN
const P3 = HIDDEN * 1
const P4 = 1
const PJ11 = diagm(P1, P_SIZE, 0 => ones(Float32, P1))
const PJ12 = diagm(P2, P_SIZE, P1 => ones(Float32, P2))
const PJ21 = diagm(P3, P_SIZE, P1 + P2 => ones(Float32, P3))
const PJ22 = diagm(P4, P_SIZE, P1 + P2 + P3 => ones(Float32, P4))

D(f) = (x, θ) -> pullback(y -> f(y, θ), x)[2](ones(Float32, 1, size(x, 2)))[1]
swift_sink(x) = x + log1p(1f5 * x) * 1f-3 # This is magic (number)
bv(i, n) = [j == i for j = 1:n]
split(phi, n) = [(x, t) -> bv(i, n)' * phi(x, t) for i = 1:n]
split(phi) = split(phi, DIM)

function build_my_model()
    _, θ1, θ2, θ3 = build_model_fast()
    ϕ(x, θ) = begin
        W1 = PJ11 * θ |> x -> reshape(x, HIDDEN, DIM)
        b1 = PJ12 * θ |> x -> reshape(x, HIDDEN, 1)
        W2 = PJ21 * θ |> x -> reshape(x, 1, HIDDEN)
        b2 = PJ22 * θ |> x -> reshape(x, 1, 1)
        W2 * (tanh.(W1 * x .+ b1)) .+ b2
    end
    (ϕ, θ1, θ2, θ3)
end


function build_model_fast()
    ϕ1 = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
    θ1 = initial_params(ϕ1)
    ϕ2 = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
    θ2 = initial_params(ϕ2)
    ϕ3 = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
    θ3 = initial_params(ϕ3)
    (ϕ1, θ1, θ2, θ3)
end

function get_domain(;show=false)
    domain = rand(Float32, DIM, BATCH_SIZE)
    if show
        zip(domain[1, :], domain[2, :]) |> collect |> scatter
    end
    domain
end

"""
f is f(x, θ)
"""
function get_loss(f)

    bsize = BATCH_SIZE |> sqrt |> ceil |> Int # boundary data batch size
    points = range(0.0f0, 1.0f0, length = bsize)'
    bd_1 = [zeros(Float32, 1, bsize); points]
    bd_2 = [ones(Float32, 1, bsize); points]
    bd_3 = [points; zeros(Float32, 1, bsize)]
    bd_4 = [points; ones(Float32, 1, bsize)]
    f_1 = zeros(Float32, 1, bsize)
    f_2 = zeros(Float32, 1, bsize)
    f_3 = zeros(Float32, 1, bsize)
    f_4 = @. sin(pi * points)
    pde_domain = get_domain()

    fx, fy = split(D(f))
    # fx = Dx(f)
    # fy = Dy(f)
    # gx = Dx(g)
    # hy = Dy(h)

    loss(θ, p) = begin
        θ1 = θ * [1.0f0; 0.0f0; 0.0f0]
        θ2 = θ * [0.0f0; 1.0f0; 0.0f0]
        θ3 = θ * [0.0f0; 0.0f0; 1.0f0]
        eq_res_1 = fx(pde_domain, θ2) + fy(pde_domain, θ3)
        eq_res_2 = fx(pde_domain, θ1) - f(pde_domain, θ2)
        eq_res_3 = fy(pde_domain, θ1) - f(pde_domain, θ3)
        eq_residual_1 = mean(swift_sink ∘ abs2, eq_res_1)
        eq_residual_2 = mean(swift_sink ∘ abs2, eq_res_2)
        eq_residual_3 = mean(swift_sink ∘ abs2, eq_res_3)
        bd_residual_1 = mean(swift_sink ∘ abs2, f(bd_1, θ1) - f_1)
        bd_residual_2 = mean(swift_sink ∘ abs2, f(bd_2, θ1) - f_2)
        bd_residual_3 = mean(swift_sink ∘ abs2, f(bd_3, θ1) - f_3)
        bd_residual_4 = mean(swift_sink ∘ abs2, f(bd_4, θ1) - f_4)
        +(
            bd_residual_1,
            bd_residual_2,
            bd_residual_3,
            bd_residual_4,
            eq_residual_1,
            eq_residual_2,
            eq_residual_3,
        )
    end
    loss_hard(θ, p) = begin
        r = loss(θ, p)
        swift_sink(r)
    end
end

function train()
    f, θ1, θ2, θ3 = build_my_model()

    opt_f = OptimizationFunction(
        get_loss(f),
        GalacticOptim.AutoZygote()
    )
    prob = OptimizationProblem(opt_f, [θ1 θ2 θ3])
    sol = solve(prob, Optim.BFGS())
    (f, sol)
end

function show_results(f, sol)
    x_test = range(0, 1, length=200)
    y_test = range(0, 1, length=200)
    func(x) = f(reshape(x, 2, 1), sol.minimizer[:, 1])[1]
    #graph_f = (x::AbstractFloat, y::AbstractFloat) -> f([x y]', sol.minimizer)[1]
    ## The array implementation is as follows, but I choose the functional way.
    graph_f = vcat.(x_test', y_test) .|> func
    p = heatmap(x_test, y_test, graph_f, aspect_ratio=:equal)
    exact(ξ) = begin
        x, y = ξ
        sin(pi * x) * sinh(pi * y) / sinh(pi)
    end
    f_error(x, p) = func(x) - exact(x) |> abs2
    qprob = QuadratureProblem(f_error, zeros(2), ones(2))
    solution = solve(qprob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
    contour!(p, x_test, y_test, graph_f,
             line=(:black),
             contour_labels=true,
             levels=10,
             colorbar_entry=false)
    annotate!(p, 0.5, 0.2, "Error = $(@sprintf("%.1e", sqrt(solution.u[1])))")
    #q = contourf(x_test, y_test, (x, y) -> f([x; y], sol.minimizer)[1])
    p
end

#f, s = train();
