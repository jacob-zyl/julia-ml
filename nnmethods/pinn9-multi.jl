# X-multi version is much faster!!!
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
#
# macro D(f)
#     return :(
#         (x, θ) -> Zygote.pullback(y -> $f(y, θ), x)[2](ones(1, size(x, 2)))[1]
#     )
# end

const DIM = 2
const HIDDEN = 10
const BATCH_SIZE = 50
const P_SIZE = DIM * HIDDEN + HIDDEN + HIDDEN * 1 + 1
const P1 = DIM * HIDDEN
const P2 = P1 + HIDDEN
const P3 = P2 + HIDDEN * 1
# function D(f)
#     grad(x, θ) = begin
#         back = Zygote.pullback(y -> f(y, θ), x)[2]
#         back(ones(1, size(x, 2)))[1]
#     end
#     # (x, θ) -> Zygote.pullback(y -> f(y, θ), x)[2](ones(1, size(x, 2)))[1]
# end
D(f) = (x, θ) -> pullback(y -> f(y, θ), x)[2](ones(Float32, 1, size(x, 2)))[1]
swift_sink(x) = x + log1p(1f5 * x) * 1f-3 # This is magic (number)
bv(i, n) = [j == i for j = 1:n]
split(phi, n) = [(x, t) -> bv(i, n)' * phi(x, t) for i = 1:n]
split(phi) = split(phi, DIM)

function build_my_model()
    θ1 = rand(Float32, P_SIZE)
    θ2 = rand(Float32, P_SIZE)
    θ3 = rand(Float32, P_SIZE)
    ϕ(x, θ) = begin
        W1 = @view θ[1:P1]
        b1 = @view θ[P1+1:P2]
        W2 = @view θ[P2+1:P3]
        b2 = @view θ[P3:end]
        W2 * (σ.(W1 * x) + b1) + b2
    end
end


function build_model()
    model1 = Chain(Dense(DIM, HIDDEN, tanh), Dense(HIDDEN, 1))
    θ1, re1 = Flux.destructure(model1)
    ϕ1(x, θ) = re1(θ)(x)
    model2 = Chain(Dense(DIM, HIDDEN, tanh), Dense(HIDDEN, 1))
    θ2, re2 = Flux.destructure(model2)
    ϕ2(x, θ) = re2(θ)(x)
    model3 = Chain(Dense(DIM, HIDDEN, tanh), Dense(HIDDEN, 1))
    θ3, re3 = Flux.destructure(model3)
    ϕ3(x, θ) = re3(θ)(x)
    (ϕ1, θ1, ϕ2, θ2, ϕ3, θ3)
end

function build_model_fast()
    ϕ1 = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
    θ1 = initial_params(ϕ1)
    ϕ2 = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
    θ2 = initial_params(ϕ2)
    ϕ3 = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
    θ3 = initial_params(ϕ3)
    (ϕ1, θ1, ϕ2, θ2, ϕ3, θ3)
end

function get_domain()
    rand(Float32, DIM, BATCH_SIZE)
    #zip(domain[1, :], domain[2, :]) |> collect |> scatter
end

"""
f is f(x, θ)
"""
function get_loss(f, g, h)

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
    gx, _ = split(D(g))
    _, hy = split(D(h))
    # fx = Dx(f)
    # fy = Dy(f)
    # gx = Dx(g)
    # hy = Dy(h)

    loss(θ, p) = begin
        θ1 = @view θ[:, 1]
        θ2 = @view θ[:, 2]
        θ3 = @view θ[:, 3]
        eq_res_1 = gx(pde_domain, θ2) + hy(pde_domain, θ3)
        eq_res_2 = fx(pde_domain, θ1) - g(pde_domain, θ2)
        eq_res_3 = fy(pde_domain, θ1) - h(pde_domain, θ3)
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
    # loss_hard(θ, p) = begin
    #     r = loss(θ, p)
    #     swift_sink(r)
    # end
end

function train()
    f, θ1, g, θ2, h, θ3 = build_model()

    opt_f = OptimizationFunction(
        get_loss(f, g, h),
        GalacticOptim.AutoZygote()
    )
    prob = OptimizationProblem(opt_f, [θ1 θ2 θ3])
    sol = solve(prob, Optim.BFGS())
    (f, sol)
end

function show_results(f, sol)
    x_test = range(0, 1, length=200)
    y_test = range(0, 1, length=200)
    graph_f = (x::AbstractFloat, y::AbstractFloat) -> f([x; y], sol.minimizer)[1]
    ## The array implementation is as follows, but I choose the functional way.
    # z = vcat.(x_test', y_test) .|> x -> f(x, s.minimizer)[1]
    # p = heatmap(x_test, y_test, z)
    p = heatmap(x_test, y_test, graph_f, aspect_ratio=:equal)
    exact(ξ) = begin
        x, y = ξ
        sin(pi * x) * sinh(pi * y) / sinh(pi)
    end
    f_error(x, p) =
        f(reshape(x, 2, 1), sol.minimizer)[1] - exact(x) |> abs2
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
