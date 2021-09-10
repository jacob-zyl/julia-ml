# Time-stamp: <2021-06-06 11:35:03 jacob>
using LinearAlgebra, StaticArrays, Statistics
using GalacticOptim, Optim
using Roots, FastGaussQuadrature

using ForwardDiff

using Printf

using Plots
pyplot()

push!(LOAD_PATH, pwd())
using FEMUtils

const nu = 0.02
_nodes, _weights = gausslegendre(4)
const nodes = _nodes
const weights = _weights
u_im(x) = (x - 1.0)/(x + 1.0) - exp(-x/nu)
const ū = find_zero(u_im, 1)

function point_residual(e, p)
    u2 = point_derivative2(e, p)
    u1 = point_derivative(e, p)
    u0 = point_interpolate(e, p)
    @. (u0 * u1 - nu * u2)^2
end

function element_loss(e; order=4)
    p = point_from_local_to_global(e, nodes)
    sum(point_residual(e, p) .* weights)
end

function train(N::Integer)
    grid = sinpi.(range(0, N, step=1) ./ 2N) |> collect
    train(grid)
end

function train(grid::Array)
    # grid = [0.0f0, 0.2f0, 0.3f0, 0.4f0, 0.6f0, 0.8f0, 0.9f0, 1.0f0]
    NN = length(grid)
    NE = NN - 1
    NK = 2
    grid_values = zeros(2, NN)
    grid_values[1] = 1.0

    function loss(grid_values, p)

        function get_value_bd0(e)
            @views [1.0; grid_values[2, 1]; grid_values[:, 2]]
        end

        function get_value_inner(e)
            @views [grid_values[:, e]; grid_values[:, e+1]]
        end

        function get_value_bd1(e)
            @views [grid_values[:, NE]; 0.0; grid_values[2, NE+1]]
        end

        function get_value(e)
            u2 = @view grid_values[:, e+1]
            if e == 1
                u = get_value_bd0(e)
            elseif e == NE
                u = get_value_bd1(e)
            else
                u = get_value_inner(e)
            end
            u
        end

        function get_value_with_penalty(e)
            get_value_inner(e)
        end

        function get_node(e)
            (@view grid[e:e+1]) |> collect
        end

        element(e) = Element(get_node(e), get_value(e))
        loss = (element_loss ∘ element).(range(1, NE, step=1)) |> sum

        # for e in range(1, NE, step=1)
        #     # element = Element(
        #     #     SVector{2, Float32}(get_node(e)),
        #     #     SVector{4, Float32}(get_value(e)))
        #     element = Element(get_node(e), get_value(e))
        #     loss = loss + element_loss(element)
        # end
    end

    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob = OptimizationProblem(opt_f, grid_values, 0)
    #prob = OptimizationProblem(loss, grid_values, 0.0f0)

    #sol = solve(prob, ADAM(), maxiters=100)
    sol = solve(prob, BFGS())
    (grid, sol, prob)
end

function show_results(grid, sol)

    # f_exact(x) = 2.0 / (1.0 + exp((x - 1.0)/nu)) - 1.0
    function f_exact(x::Real)
        ū * (2/(1 + exp(ū*(x-1)/nu)) - 1)
    end

    function fx_exact(x::Real)
        ForwardDiff.derivative(f_exact, x)
    end


    xgrid = range(0, 1, length=100)

    p = plot(xgrid, f_exact.(xgrid), label="analytical")
    scatter!(p, grid, sol.minimizer[1, :], label="simulated")
    err1 = mean(abs2, f_exact.(grid) - sol.minimizer[1, :]) |> sqrt
    annotate!(p, 0.5, 0.25, "L2 Error = $(@sprintf("%.1e", err1))")
    annotate!(p, 0.5, 0.4, "Re = $(@sprintf("%7.1e", nu^-1))")
    annotate!(p, 0.5, 0.5, "Number of Nodes = $(@sprintf("%2i", size(grid, 1)))")
    title!(p, "Function Value on Chebyshev Grid")

    q = plot(xgrid, fx_exact.(xgrid), label="analytical")
    scatter!(q, grid, sol.minimizer[2, :], label="simulated")
    title!(q, "Derivative Value")
    err2 = mean(abs2, fx_exact.(grid) - sol.minimizer[2, :])
    annotate!(q, 0.5, -30, "L2 Error = $(@sprintf("%.1e", err2))")

    plot(p, q, layout=(1, 2), size=(1000, 300), legend=:bottomleft)
end

function show_results(result)
    show_results(result[1], result[2])
end
