using LinearAlgebra, Statistics, StaticArrays
using Flux, DiffEqFlux
using GalacticOptim, Optim
using Plots
plotly()
theme(:vibrant)

push!(LOAD_PATH, pwd())
using Utils

### CONSTANTS
###
### Independent constants
const DIM = 1
const DIM_OUT = 1
const HIDDEN = 5
const BATCH_SIZE = 20
### Dependent constants
### END


function get_training_set()
    get_domain(DIM, BATCH_SIZE)
end

function build_my_model(hidden=HIDDEN)

    n = range(1.0f0, step=1.0f0, length=hidden-1) |> collect
    base(x::Matrix{Float32}) = begin
        @. exp(n * log(x))
    end

    ### CONSTANTS
    ###
    ### Dependent constants
    P_SIZE = DIM * HIDDEN + HIDDEN + HIDDEN * 1 + 1
    P1 = DIM * HIDDEN
    P2 = HIDDEN
    P3 = HIDDEN * DIM_OUT
    P4 = DIM_OUT
    PJ11 = diagm(P1, P_SIZE, 0 => ones(Float32, P1))
    PJ12 = diagm(P2, P_SIZE, P1 => ones(Float32, P2))
    PJ21 = diagm(P3, P_SIZE, P1 + P2 => ones(Float32, P3))
    PJ22 = diagm(P4, P_SIZE, P1 + P2 + P3 => ones(Float32, P4))
    ### END
    function build_model_fast()
        network = FastChain(FastDense(DIM, HIDDEN, tanh), FastDense(HIDDEN, 1))
        θ1 = initial_params(network)
        θ2 = initial_params(network)
        (network, θ1, θ2)
    end
    ϕ_ann(θ, x) = begin
        W1 = reshape(PJ11 * θ, HIDDEN, DIM)
        b1 = reshape(PJ12 * θ, HIDDEN, 1)
        W2 = reshape(PJ21 * θ, 1, HIDDEN)
        b2 = reshape(PJ22 * θ, 1, 1)

        W2 * (sin.(W1 * x .+ b1)) .+ b2
    end
    _, θ1_ann, θ2_ann = build_model_fast()
    ϕ_taylor(θ, x) = begin
        ## split the parameters first
        a0 = diagm(1, hidden, 0 => ones(Float32, 1)) * θ
        a_rst = diagm(hidden-1, hidden, 1 => ones(Float32, hidden-1)) * θ
        x .* (1.0f0 .- x) .* (a_rst' * base(x)) + x .* (1.0f0 .- x) .* a0
    end
    θ1_taylor = zeros(Float32, hidden)
    θ2_taylor = zeros(Float32, hidden)
    # #if method == :ann
    #     ϕ = ϕ_ann
    #     θ1 = θ1_ann
    #     θ2 = θ2_ann
    #elseif method == :taylor
        ϕ = ϕ_taylor
        θ1 = θ1_taylor
        θ2 = θ2_taylor
    #end
    (ϕ, θ1, θ2)
end

function get_loss(ϕ)
    # reduce_func = cliff ∘ abs2
    reduce_func = abs2
    loss(θ, p) = begin
        u = θ * [1.0f0; 0.0f0]
        v = θ * [0.0f0; 1.0f0]
        domain = p
        Du = D(ϕ, u, domain)
        Dv = D(ϕ, v, domain)
        Dϕ = Df(ϕ)
        DDϕ = Df(Dϕ)

        +(
            mean(reduce_func, DDϕ(u, domain) + ϕ(u, domain) + domain),
            # mean(reduce_func, Du - ϕ(v, domain)),
            mean(reduce_func, ϕ(u, [1.0f0 0.0f0]) - [0.0f0 0.0f0]),
        )
    end
end

function train(; kwargs...)
    ϕ, u, v = build_my_model()
    train(ϕ, [u v]; kwargs...)
end

function train(ϕ, Θ::Array; optimizer=ADAM(), maxiters=500)
    domain = get_training_set()
    opt_f = OptimizationFunction(get_loss(ϕ), GalacticOptim.AutoForwardDiff())
    prob = OptimizationProblem(opt_f, Θ, domain)
    sol = solve(prob, optimizer; maxiters=maxiters)
    (ϕ, sol)
end

function train(ϕ, sol::Optim.MultivariateOptimizationResults; kwargs...)
    train(ϕ, sol.minimizer; kwargs...)
end

function train(ϕ, sol::SciMLBase.OptimizationSolution; kwargs...)
    train(ϕ, sol.minimizer; kwargs...)
end

function show_results(ϕ, sol)
    θ = sol.minimizer * [1.0f0; 0.0f0]
    x_test = reshape(range(0.0f0, 1.0f0, length=77) |> collect, 1, :)
    f_exact(x) = sin(x)/sin(1.0f0) - x
    #f_exact(x) = x * (1.0f0 - x)
    p = plot(x_test', f_exact.(x_test'))
    plot!(p, x_test', ϕ(θ, x_test)',
          linestyle=:dot)
    q = plot(x_test', f_exact.(x_test') - ϕ(θ, x_test)')
    plot(p, q)
end
