# not really working
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
const BATCH_SIZE = 100
const PARAM_LEN = DIM * HIDDEN + HIDDEN + HIDDEN + 1
D(f) = (x, θ) -> pullback(y -> f(y, θ), x)[2](ones(1, size(x, 2)))[1]
swift_sink(x) = x + log1p(1f5 * x) * 1f-3 # This is magic (number)
bv(i, n) = [j == i for j = 1:n]
split(phi, n) = [(x, t) -> bv(i, n)' * phi(x, t) for i = 1:n]
split(phi) = split(phi, DIM)

function build_model()
    model1 = Chain(Dense(DIM, HIDDEN, tanh), Dense(HIDDEN, 1))
    model2 = Chain(Dense(DIM, HIDDEN, tanh), Dense(HIDDEN, 1))
    model3 = Chain(Dense(DIM, HIDDEN, tanh), Dense(HIDDEN, 1))
    (model1, model2, model3)
end

function get_domain()
    rand(Float32, DIM, BATCH_SIZE)
    #zip(domain[1, :], domain[2, :]) |> collect |> scatter
end

"""
f is f(x, θ)
"""
function get_loss(ff, gg, hh)

    pde_domain = get_domain()

    loss(θ, p) = begin
        θ1 = @view θ[1:PARAM_LEN]
        θ2 = @view θ[PARAM_LEN+1:2PARAM_LEN]
        θ3 = @view θ[2PARAM_LEN+1:end]
        l = BATCH_SIZE

        fx = pushforward(ff(θ1), pde_domain)([ones(l) zeros(l)]')
        fy = pushforward(ff(θ1), pde_domain)([zeros(l) ones(l)]')
        gx = pushforward(gg(θ2), pde_domain)([ones(l) zeros(l)]')
        hy = pushforward(hh(θ3), pde_domain)([zeros(l) ones(l)]')

        eq_res_1 = gx + hy
        eq_res_2 = fx - gg(θ2)(pde_domain)
        eq_res_3 = fy - hh(θ3)(pde_domain)
        eq_residual_1 = mean(swift_sink ∘ abs2, eq_res_1)
        eq_residual_2 = mean(swift_sink ∘ abs2, eq_res_2)
        eq_residual_3 = mean(swift_sink ∘ abs2, eq_res_3)
        +(
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

function get_cons(ff, gg, hh)
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
    loss(θ, p) = begin
        θ1 = @view θ[1:PARAM_LEN]
        θ2 = @view θ[PARAM_LEN+1:2PARAM_LEN]
        θ3 = @view θ[2PARAM_LEN+1:end]
        l = BATCH_SIZE

        bd_residual_1 = mean(swift_sink ∘ abs2, ff(θ1)(bd_1) - f_1)
        bd_residual_2 = mean(swift_sink ∘ abs2, ff(θ1)(bd_2) - f_2)
        bd_residual_3 = mean(swift_sink ∘ abs2, ff(θ1)(bd_3) - f_3)
        bd_residual_4 = mean(swift_sink ∘ abs2, ff(θ1)(bd_4) - f_4)
        +(bd_residual_1,
          bd_residual_2,
          bd_residual_3,
          bd_residual_4)
    end
    (θ, p) -> [loss(θ, p)]
end

function train()
    modelf, modelg, modelh = build_model()
    θ1, f = Flux.destructure(modelf)
    θ2, g = Flux.destructure(modelg)
    θ3, h = Flux.destructure(modelh)

    opt_f = OptimizationFunction(
        get_loss(f, g, h),
        GalacticOptim.AutoForwardDiff();
        cons=get_cons(f, g, h))
    prob = OptimizationProblem(opt_f, [θ1; θ2; θ3])
    sol = solve(prob, Newton())
    ((x, t) -> f(t)(x), sol)
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
