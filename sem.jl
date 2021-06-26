using LinearAlgebra, StaticArrays, Statistics

using GalacticOptim, Optim
using Roots, FastGaussQuadrature

using ForwardDiff

using Printf

using Plots
pyplot()

const NK = 4                    # element DOF
const nu = 0.1
_nodes, _weights = gausslobatto(NK)
const nodes = _nodes
const weights = _weights
const ū = find_zero(x -> (x - 1.0)/(x + 1.0) - exp(-x/nu), 1)

# f_exact(x) = 2.0 / (1.0 + exp((x - 1.0)/nu)) - 1.0
function f_exact(x::Real)
    ū * (2/(1 + exp(ū*(x-1)/nu)) - 1)
end

function fx_exact(x::Real)
    ForwardDiff.derivative(f_exact, x)
end

struct Element{T<:Real}
    node::Vector{T}
    data::Vector{T}
end
Element(node::Vector{T}) where {T} = Element(node, zeros(T, NK))
# struct Element{T<:Real}
#     node::SVector{2, T}
#     data::SVector{4, T}
# end

function area(el::Element{T}) where {T}
    el.node[end] - el.node[1]
end

function jacobian(el::Element{T}) where {T}
    2one(T) / area(el)
end

function jacobianr(el::Element{T}) where {T}
    area(el) / 2one(T)
end

function point_from_global_to_local(el::Element, p)
    """
    Currently this function is foundamentally flaw.
    """
    x = el.node
    k = 2.0 / (x[end] - x[1])
    b = (x[1] + x[end]) / (x[1] - x[end])
    @. k * p + b
end

function point_from_local_to_global(el::Element, p)
    x = el.node
    k = (x[end] - x[1]) * 0.5
    b = (x[end] + x[1]) * 0.5
    @. k * p + b
end

# Definition of Lagrangian polynomials
H1(x) = @. (x - nodes[2]) * (x - nodes[3]) * (x - nodes[4]) / ((nodes[1] - nodes[2]) * (nodes[1] - nodes[3]) * (nodes[1] - nodes[4]))
H2(x) = @. (x - nodes[3]) * (x - nodes[4]) * (x - nodes[1]) / ((nodes[2] - nodes[3]) * (nodes[2] - nodes[4]) * (nodes[2] - nodes[1]))
H3(x) = @. (x - nodes[4]) * (x - nodes[1]) * (x - nodes[2]) / ((nodes[3] - nodes[4]) * (nodes[3] - nodes[1]) * (nodes[3] - nodes[2]))
H4(x) = @. (x - nodes[1]) * (x - nodes[2]) * (x - nodes[3]) / ((nodes[4] - nodes[1]) * (nodes[4] - nodes[2]) * (nodes[4] - nodes[3]))

h1(x) = @. ((x - nodes[2]) * (x - nodes[3]) + (x - nodes[3]) * (x - nodes[4]) + (x - nodes[4]) * (x - nodes[2])) / ((nodes[1] - nodes[2]) * (nodes[1] - nodes[3]) * (nodes[1] - nodes[4]))
h2(x) = @. ((x - nodes[3]) * (x - nodes[4]) + (x - nodes[4]) * (x - nodes[1]) + (x - nodes[1]) * (x - nodes[3])) / ((nodes[2] - nodes[3]) * (nodes[2] - nodes[4]) * (nodes[2] - nodes[1]))
h3(x) = @. ((x - nodes[4]) * (x - nodes[1]) + (x - nodes[1]) * (x - nodes[2]) + (x - nodes[2]) * (x - nodes[4])) / ((nodes[3] - nodes[4]) * (nodes[3] - nodes[1]) * (nodes[3] - nodes[2]))
h4(x) = @. ((x - nodes[1]) * (x - nodes[2]) + (x - nodes[2]) * (x - nodes[3]) + (x - nodes[3]) * (x - nodes[1])) / ((nodes[4] - nodes[1]) * (nodes[4] - nodes[2]) * (nodes[4] - nodes[3]))

hh1(x) = @. 2.0 * ((x - nodes[2]) + (x - nodes[3]) + (x - nodes[4])) / ((nodes[1] - nodes[2]) * (nodes[1] - nodes[3]) * (nodes[1] - nodes[4]))
hh2(x) = @. 2.0 * ((x - nodes[3]) + (x - nodes[4]) + (x - nodes[1])) / ((nodes[2] - nodes[3]) * (nodes[2] - nodes[4]) * (nodes[2] - nodes[1]))
hh3(x) = @. 2.0 * ((x - nodes[4]) + (x - nodes[1]) + (x - nodes[2])) / ((nodes[3] - nodes[4]) * (nodes[3] - nodes[1]) * (nodes[3] - nodes[2]))
hh4(x) = @. 2.0 * ((x - nodes[1]) + (x - nodes[2]) + (x - nodes[3])) / ((nodes[4] - nodes[1]) * (nodes[4] - nodes[2]) * (nodes[4] - nodes[3]))

H(x) = [H1(x) H2(x) H3(x) H4(x)]
h(x) = [h1(x) h2(x) h3(x) h4(x)]
hh(x) = [hh1(x) hh2(x) hh3(x) hh4(x)]


function point_interpolate(el, p)
    H(p) * el.data
end

function point_derivative(el, p)
    h(p) * el.data .* jacobian(el)
end

function point_derivative2(el, p)
    hh(p) * el.data .* jacobian(el)^2
end

function point_residual(el, p)
    u2 = point_derivative2(el, p)
    u1 = point_derivative(el, p)
    u0 = point_interpolate(el, p)
    j = jacobian(el)
    # This should be element equation with local abscissae instead of global abscissae
    res = @. u0 * u1 - nu * u2
    # end of element equation
    res.^2
end

function element_loss(el::Element)
    p = nodes
    sum(point_residual(el, p) .* weights) * jacobianr(el)
end

loss(node_values, grid_points) = begin
    NE = size(grid_points, 1) - 1
    NN = 3NE + 1
    get_node(ne) = @views grid_points[ne:ne+1] |> collect
    get_value_bd0(ne) = @views [1.0 ; node_values[3ne-1 : 3ne+1]]
    get_value_inner(ne) = @views node_values[3ne - 2 : 3ne + 1] |> collect
    get_value_bd1(ne) = @views [node_values[3ne - 2 : 3ne] ; 0.0]
    get_value(ne) = begin
        if ne == 1
            u = get_value_bd0(ne)
        elseif ne == NE
            u = get_value_bd1(ne)
        else
            u = get_value_inner(ne)
        end
        u
    end
    get_element(ne) = Element(get_node(ne), get_value(ne))
    (element_loss ∘ get_element).(range(1, NE, step=1)) |> sum
end

function train(grid::Array)
    NE = size(grid, 1) - 1
    NN = 3NE + 1
    node_values = zeros(NN)
    node_values[1] = 1.0

    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob = OptimizationProblem(opt_f, node_values, grid)
    sol = solve(prob, ADAM(), maxiters=100)
    (grid, sol, prob)
end

function train(N::Integer)
    grid = sinpi.(range(0, N, step=1) ./ 2N)
    train(grid)
end

function show_results(grid, sol)

    n = size(grid, 1)
    f_simu = sol.minimizer[3 .* (0:n-1) .+ 1]

    xgrid = range(0, 1, length=100)

    p = plot(xgrid, f_exact.(xgrid), label="analytical")
    scatter!(p, grid, f_simu, label="simulated")
    err1 = mean(abs2, f_exact.(grid) - f_simu) |> sqrt
    annotate!(p, 0.5, 0.25, "L2 Error = $(@sprintf("%.1e", err1))")
    annotate!(p, 0.5, 0.4, "Re = $(@sprintf("%7.1e", nu))")
    annotate!(p, 0.5, 0.5, "Number of Nodes = $(@sprintf("%2i", size(grid, 1)))")
    title!(p, "Function Value on Chebyshev Grid")

    # q = plot(xgrid, fx_exact.(xgrid), label="analytical")
    # scatter!(q, grid, fx_simu, label="simulated")
    # title!(q, "Derivative Value")
    # err2 = mean(abs2, fx_exact.(grid) - fx_simu)
    # annotate!(q, 0.5, -30, "L2 Error = $(@sprintf("%.1e", err2))")

    # plot(p, q, layout=(1, 2), size=(1000, 300), legend=:bottomleft)
    p
end

function show_results(result)
    show_results(result[1], result[2])
end
