using LinearAlgebra
using Statistics: mean
using Flux
using Zygote
using GalacticOptim
using Optim
using Quadrature
using Printf
using Plots
plotly()
theme(:vibrant)
#
# macro D(f)
#     return :(
#         (x, θ) -> Zygote.pullback(y -> $f(y, θ), x)[2](ones(1, size(x, 2)))[1]
#     )
# end

const DIM = 2
const BATCH_SIZE = 100
function D(f)
    return (x, θ) -> Zygote.pullback(y -> f(y, θ), x)[2](ones(1, size(x, 2)))[1]
end
swift_sink(x) = x + log1p(1f5 * x) * 1f-3 # This is magic (number)
bv(i, n) = [j == i for j = 1:n]
split(phi, n) = [(x, t) -> bv(i, n)' * phi(x, t) for i = 1:n]
split(phi) = split(phi, DIM)

function build_model()
    model1 = Chain(Dense(DIM, 20, tanh), Dense(20, 1))
    θ1, re1 = Flux.destructure(model1)
    ϕ1(x, θ) = re1(θ)(x)
    model2 = Chain(Dense(DIM, 20, tanh), Dense(20, 1))
    θ2, re2 = Flux.destructure(model2)
    ϕ2(x, θ) = re2(θ)(x)
    model3 = Chain(Dense(DIM, 20, tanh), Dense(20, 1))
    θ3, re3 = Flux.destructure(model3)
    ϕ3(x, θ) = re3(θ)(x)
    (ϕ1, θ1, re1, ϕ2, θ2, re2, ϕ3, θ3, re3)
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
    bd_1 = [zeros(1, bsize); points]
    bd_2 = [ones(1, bsize); points]
    bd_3 = [points; zeros(1, bsize)]
    bd_4 = [points; ones(1, bsize)]
    f_1 = zeros(1, bsize)
    f_2 = zeros(1, bsize)
    f_3 = zeros(1, bsize)
    f_4 = @. sin(pi * points)
    pde_domain = get_domain()

    fx, fy = split(D(f))
    gx, _ = split(D(g))
    _, hy = split(D(h))

    loss(θ, p) = begin
        θ1 = @view θ[:, 1]
        θ2 = @view θ[:, 2]
        θ3 = @view θ[:, 3]
        eq_res_1 = gx(pde_domain, θ2) + hy(pde_domain, θ3)
        eq_res_2 = fx(pde_domain, θ1) - g(pde_domain, θ2)
        eq_res_3 = fy(pde_domain, θ1) - h(pde_domain, θ3)
        eq_residual_1 = mean(abs2, eq_res_1)
        eq_residual_2 = mean(abs2, eq_res_2)
        eq_residual_3 = mean(abs2, eq_res_3)
        bd_residual_1 = mean(abs2, f(bd_1, θ) - f_1)
        bd_residual_2 = mean(abs2, f(bd_2, θ) - f_2)
        bd_residual_3 = mean(abs2, f(bd_3, θ) - f_3)
        bd_residual_4 = mean(abs2, f(bd_4, θ) - f_4)
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
    f, θ1, re1, g, θ2, re2, h, θ3, re3 = build_model()

    opt_f = OptimizationFunction(
        get_loss(f, g, h),
        GalacticOptim.AutoForwardDiff()
    )
    prob = OptimizationProblem(opt_f, [θ1 θ2 θ3])
    sol = solve(prob, Optim.BFGS())

    (f, sol)
end

function show_results(f, sol)
    x_test = range(0, 1, length=200)
    y_test = range(0, 1, length=200)
    graph_f = (x, y) -> f([x; y], sol.minimizer)[1]
    ## The array implementation is as follows, but I choose the functional way.
    # z = vcat.(x_test', y_test) .|> x -> f(x, s.minimizer)[1]
    # p = heatmap(x_test, y_test, z)
    p = heatmap(x_test, y_test, graph_f,
                aspect_ratio=:equal)
    exact(x) = begin
        sin(pi * x[1]) * sinh(pi * x[2]) / sinh(pi)
    end
    f_error(x, p) = f([x[1]; x[2]], sol.minimizer) .- exact(x) .|> abs2
    qprob = QuadratureProblem(f_error, zeros(2), ones(2))
    solution = solve(qprob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
    contour!(p, x_test, y_test, graph_f,
             line=(:dot, :arrow, :black),
             contour_labels=true,
             levels=10,
             colorbar_entry=false)
    annotate!(p, 0.5, 0.2, "Error = $(@sprintf("%.1e", solution.u[1]))")
    #q = contourf(x_test, y_test, (x, y) -> f([x; y], sol.minimizer)[1])
    p
end

@fastmath @inbounds f, s = train()
