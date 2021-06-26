# Time-stamp: <2021-06-06 11:35:03 jacob>
using LinearAlgebra, StaticArrays, Statistics
using GalacticOptim, Optim

using Printf

using Plots
pyplot()

# struct Element{T<:Real}
#     node::SVector{2, T}
#     data::SVector{4, T}
# end

struct Element{T<:Real}
    node::Vector{T}
    data::Vector{T}
end

function area(e::Element{T}) where {T}
    e.node[2] - e.node[1]
end

function jacobian(e::Element{T}) where {T}
    2one(T) / area(e)
end

function jacobianr(e::Element{T}) where {T}
    area(e) / 2one(T)
end


function point_from_global_to_local(e::Element, p)
    x = e.node
    k = 2.0f0 / (x[2] - x[1])
    b = (x[1] + x[2]) / (x[1] - x[2])
    @. k * p + b
end

function point_from_local_to_global(e::Element, p)
    x = e.node
    k = (x[2] - x[1]) * 0.5f0
    b = (x[2] + x[1]) * 0.5f0
    @. k * p + b
end

H1(x) = @. 0.25f0 * (1.0f0 - x)^2 * (2.0f0 + x)
H2(x) = @. 0.25f0 * (1.0f0 - x)^2 * (x + 1.0f0)
H3(x) = @. 0.25f0 * (1.0f0 + x)^2 * (2.0f0 - x)
H4(x) = @. 0.25f0 * (1.0f0 + x)^2 * (x - 1.0f0)
h1(x) = @. 0.75f0 * (x^2 - 1.0f0)
h2(x) = @. 0.25f0 * (3.0f0x^2 - 2.0f0x - 1.0f0)
h3(x) = @. -0.75f0 * (x^2 - 1.0f0)
h4(x) = @. 0.25f0 * (3.0f0x^2 + 2.0f0x - 1.0f0)
hh1(x) = @. 1.5f0x
hh2(x) = @. 1.5f0x - 0.5f0
hh3(x) = @. -1.5f0x
hh4(x) = @. 1.5f0x + 0.5f0

H1(e, x) = H1(point_from_global_to_local(e, x))
H2(e, x) = H2(point_from_global_to_local(e, x)) .* jacobianr(e)
H3(e, x) = H3(point_from_global_to_local(e, x))
H4(e, x) = H4(point_from_global_to_local(e, x)) .* jacobianr(e)

h1(e, x) = h1(point_from_global_to_local(e, x)) .* jacobian(e)
h2(e, x) = h2(point_from_global_to_local(e, x))
h3(e, x) = h3(point_from_global_to_local(e, x)) .* jacobian(e)
h4(e, x) = h4(point_from_global_to_local(e, x))

hh1(e, x) = hh1(point_from_global_to_local(e, x)) .* jacobian(e)^2
hh2(e, x) = hh2(point_from_global_to_local(e, x)) .* jacobian(e)
hh3(e, x) = hh3(point_from_global_to_local(e, x)) .* jacobian(e)^2
hh4(e, x) = hh4(point_from_global_to_local(e, x)) .* jacobian(e)

H(e, x) = [H1(e, x) H2(e, x) H3(e, x) H4(e, x)]
h(e, x) = [h1(e, x) h2(e, x) h3(e, x) h4(e, x)]
hh(e, x) = [hh1(e, x) hh2(e, x) hh3(e, x) hh4(e, x)]


function point_interpolate(e, p)
    H(e, p) * e.data
end

function point_derivative(e, p)
    h(e, p) * e.data
end

function point_derivative2(e, p)
    hh(e, p) * e.data
end

function point_residual(e, p)
    u2 = point_derivative2(e, p)
    # u1 = point_derivative(e, p)
    u0 = point_interpolate(e, p)
    #b = point_interpolate_local(u, p)
    @. (u2 + u0 + p)^2
end

function element_loss(e; order=4)
    ## two points
    # p = [0.5773503f0, -0.5773503f0]
    # w = [1.0f0, 1.0f0]

    # four points
    p_local = [-0.8611363f0, -0.3399810f0, 0.3399810f0, 0.8611363f0]
    w = [0.3478548f0, 0.6521452f0, 0.6521452f0, 0.3478548f0]
    p = point_from_local_to_global(e, p_local)
    sum(point_residual(e, p) .* w)
end

function train(N::Integer)
    train(range(0.0f0, 1.0f0, length=N) |> collect)
end

function train(grid::Array)
    # grid = [0.0f0, 0.2f0, 0.3f0, 0.4f0, 0.6f0, 0.8f0, 0.9f0, 1.0f0]
    NN = length(grid)
    NE = NN - 1
    NK = 2
    grid_values = zeros(Float32, 2, NN)

    function loss(grid_values, p)

        function get_value_bd0(e)
            @views [0.0f0; grid_values[2, 1]; grid_values[:, 2]]
        end

        function get_value_inner(e)
            @views [grid_values[:, e]; grid_values[:, e+1]]
        end

        function get_value_bd1(e)
            @views [grid_values[:, NE]; 0.0f0; grid_values[2, NE+1]]
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

        function get_node(e)
            (@view grid[e:e+1]) |> collect
        end

        loss = 0.0f0

        for e in range(1, NE, step=1)
            # element = Element(
            #     SVector{2, Float32}(get_node(e)),
            #     SVector{4, Float32}(get_value(e)))
            element = Element(get_node(e), get_value(e))
            loss = loss + element_loss(element)
        end

        loss
    end

    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob = OptimizationProblem(opt_f, grid_values, 0.0f0)
    #prob = OptimizationProblem(loss, grid_values, 0.0f0)

    #sol = solve(prob, ADAM(), maxiters=100)
    sol = solve(prob, BFGS())
    (grid, sol)
end

function show_results(grid, sol)
    xgrid = range(0, 1, length=100)
    f_exact(x) = sin(x) / sin(1) - x
    fx_exact(x) = cos(x) / sin(1) - 1

    p = plot(xgrid, f_exact.(xgrid), label="analytical")
    scatter!(p, grid, sol.minimizer[1, :], label="simulated")
    title!(p, "Function Value")
    err1 = mean(abs2, f_exact.(grid) - sol.minimizer[1, :])
    annotate!(p, 0.5, 0.01, "L2 Error = $(@sprintf("%.1e", err1))")


    q = plot(xgrid, cos.(xgrid)/sin(1) .- 1, label="analytical")
    scatter!(q, grid, sol.minimizer[2, :], label="simulated")
    title!(q, "Derivative Value")
    err2 = mean(abs2, fx_exact.(grid) - sol.minimizer[1, :])
    annotate!(q, 0.5, -0.3, "L2 Error = $(@sprintf("%.1e", err2))")

    plot(p, q, layout=(1, 2), size=(1000, 300))
end

function show_results(result)
    show_results(result[1], result[2])
end
