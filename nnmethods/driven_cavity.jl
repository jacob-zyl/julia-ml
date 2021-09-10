using LinearAlgebra, Statistics, StaticArrays
using Flux, DiffEqFlux
using GalacticOptim, Optim
using Plots
plotlyjs()
theme(:vibrant)

push!(LOAD_PATH, pwd())
using Utils
using ConstantsDrivenCavity

function build_model_fast()
    network = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
    θ1 = initial_params(network)
    θ2 = initial_params(network)
    θ3 = initial_params(network)
    θ4 = initial_params(network)
    θ5 = initial_params(network)
    θ6 = initial_params(network)
    (network, θ1, θ2, θ3, θ4, θ5, θ6)
end

function build_my_model()
    ϕ(θ, x) = begin
        W1 = reshape(PJ11 * θ, HIDDEN, DIM)
        b1 = reshape(PJ12 * θ, HIDDEN, 1)
        W2 = reshape(PJ21 * θ, 1, HIDDEN)
        b2 = reshape(PJ22 * θ, 1, 1)

        W2 * (tanh.(W1 * x .+ b1)) .+ b2
    end
    _, θ1, θ2, θ3, θ4, θ5, θ6 = build_model_fast()
    (ϕ, θ1, θ2, θ3, θ4, θ5, θ6)
end

function get_loss(ϕ)
    # Figure of configuration:
    #
    #             bd_4
    #            ---->
    #     +-------------------+
    #     |                   |
    #     |                   |
    #     |                   |
    #     |                   |
    # bd_1|                   | bd_2
    #     |                   |
    #     |                   |
    #     |                   |
    #     |                   |
    #     +-------------------+
    #             bd_3

    # f is stream function ψ
    # g = ∂f / ∂y
    # h = ∂f / ∂x
    # a = ∂g / ∂y + ∂h / ∂x
    # b = ∂a / ∂x
    # c = ∂a / ∂y
    # g b - h c = ν( ∂b / ∂x + ∂c / ∂y )
    #
    points = range(0.0f0, 1.0f0, length = BD_SIZE)' # x or y

    bd_1 = [zeros(Float32, 1, BD_SIZE); points] # x = 0, y = 0..1
    bd_2 = [ones(Float32, 1, BD_SIZE); points]  # x = 1, y = 0..1
    bd_3 = [points; zeros(Float32, 1, BD_SIZE)] # x = 0..1, y = 0
    bd_4 = [points; ones(Float32, 1, BD_SIZE)]  # x = 0..1, y = 1

    # this is b.c. for stream function
    f_1 = zeros(Float32, 1, BD_SIZE)
    f_2 = zeros(Float32, 1, BD_SIZE)
    f_3 = zeros(Float32, 1, BD_SIZE)
    f_4 = zeros(Float32, 1, BD_SIZE)

    # this is b.c. for u
    g_1 = zeros(Float32, 1, BD_SIZE)
    g_2 = zeros(Float32, 1, BD_SIZE)
    g_3 = zeros(Float32, 1, BD_SIZE)
    # u = 16 x^2 ( 1 - x^2 )
    g_4 = @. 16.0f0 * points^2 * (1.0f0 - points^2)

    # this is b.c. for -v
    h_1 = zeros(Float32, 1, BD_SIZE)
    h_2 = zeros(Float32, 1, BD_SIZE)
    h_3 = zeros(Float32, 1, BD_SIZE)
    h_4 = zeros(Float32, 1, BD_SIZE)

    loss(θ, pde_domain) = begin
        # Here, θ is a N×6 matrix. Each sub-matrix of N×1 is for one variable.
        # To obtain one variable's individual parameters, just right multiply θ
        # by elementary column vector.
        f = θ * [1.0f0; 0.0f0; 0.0f0; 0.0f0; 0.0f0; 0.0f0]
        g = θ * [0.0f0; 1.0f0; 0.0f0; 0.0f0; 0.0f0; 0.0f0]
        h = θ * [0.0f0; 0.0f0; 1.0f0; 0.0f0; 0.0f0; 0.0f0]
        b = θ * [0.0f0; 0.0f0; 0.0f0; 1.0f0; 0.0f0; 0.0f0]
        c = θ * [0.0f0; 0.0f0; 0.0f0; 0.0f0; 1.0f0; 0.0f0]
        a = θ * [0.0f0; 0.0f0; 0.0f0; 0.0f0; 0.0f0; 1.0f0]
        reduce_func = cliff ∘ abs2

        f1 = ϕ(f, pde_domain)
        g1 = ϕ(g, pde_domain)
        h1 = ϕ(h, pde_domain)
        a1 = ϕ(a, pde_domain)
        b1 = ϕ(b, pde_domain)
        c1 = ϕ(c, pde_domain)

        Dxf, Dyf = getxy(D(ϕ, f, pde_domain))
        _, Dyg = getxy(D(ϕ, g, pde_domain))
        Dxh, _ = getxy(D(ϕ, h, pde_domain))
        Dxa, Dya = getxy(D(ϕ, a, pde_domain))
        Dxb, _ = getxy((D(ϕ, b, pde_domain)))
        _, Dyc = getxy((D(ϕ, c, pde_domain)))

        f21 = ϕ(f, bd_1)
        f22 = ϕ(f, bd_2)
        f23 = ϕ(f, bd_3)
        f24 = ϕ(f, bd_4)

        g21 = ϕ(g, bd_1)
        g22 = ϕ(g, bd_2)
        g23 = ϕ(g, bd_3)
        g24 = ϕ(g, bd_4)

        h21 = ϕ(h, bd_1)
        h22 = ϕ(h, bd_2)
        h23 = ϕ(h, bd_3)
        h24 = ϕ(h, bd_4)

        eq_res_1 = g1 - Dyf
        eq_res_2 = h1 - Dxf
        eq_res_3 = Dyg + Dxh - a1
        eq_res_4 = b1 - Dxa
        eq_res_5 = c1 - Dya
        eq_res_6 = @. g1 * b1 - h1 * c1 - NU * (Dxb + Dyc)

        bd_res_f1 = f21 - f_1
        bd_res_f2 = f22 - f_2
        bd_res_f3 = f23 - f_3
        bd_res_f4 = f24 - f_4

        bd_res_g1 = g21 - g_1
        bd_res_g2 = g22 - g_2
        bd_res_g3 = g23 - g_3
        bd_res_g4 = g24 - g_4

        bd_res_h1 = h21 - h_1
        bd_res_h2 = h22 - h_2
        bd_res_h3 = h23 - h_3
        bd_res_h4 = h24 - h_4

        +( ## mean works here while sum does not lol
            mean(reduce_func, eq_res_1),
            mean(reduce_func, eq_res_2),
            mean(reduce_func, eq_res_3),
            mean(reduce_func, eq_res_4),
            mean(reduce_func, eq_res_5),
            mean(reduce_func, eq_res_6),
            mean(reduce_func, bd_res_f1),
            mean(reduce_func, bd_res_f2),
            mean(reduce_func, bd_res_f3),
            mean(reduce_func, bd_res_f4),
            mean(reduce_func, bd_res_g1),
            mean(reduce_func, bd_res_g2),
            mean(reduce_func, bd_res_g3),
            mean(reduce_func, bd_res_g4),
            mean(reduce_func, bd_res_h1),
            mean(reduce_func, bd_res_h2),
            mean(reduce_func, bd_res_h3),
            mean(reduce_func, bd_res_h4))
    end
    # loss_hard(θ, p) = begin
    #     r = loss(θ, p)
    #     cliff(r)
    # end
end

function train(ϕ, Θ::Array; optimizer=ADAM(), maxiters=500)
    pde_domain = get_domain(DIM, BATCH_SIZE)
    opt_f = OptimizationFunction(get_loss(ϕ), GalacticOptim.AutoZygote())
    prob = OptimizationProblem(opt_f, Θ, pde_domain)
    sol = solve(prob, optimizer; maxiters=maxiters)
    (ϕ, sol)
end

function train(ϕ, sol::Optim.MultivariateOptimizationResults; kwargs...)
    train(ϕ, sol.minimizer; kwargs...)
end

function train(ϕ, sol::SciMLBase.OptimizationSolution; kwargs...)
    train(ϕ, sol.minimizer; kwargs...)
end

function train(; kwargs...)
    ϕ, f, g, h, b, c, a = build_my_model()
    train(ϕ, [f g h b c a]; kwargs...)
end

# ## Profiling
# using Profile
# f, s = train()

# Profile.clear()
# @profile train()
# Juno.profiler(; C=true)
# @profiler train() combine=true
