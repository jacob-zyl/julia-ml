# The correct SEM
#

using LinearAlgebra, StaticArrays, Statistics

using GalacticOptim, Optim
using Roots, FastGaussQuadrature

using ForwardDiff

using Printf

using Plots
pyplot()

const nu = 0.1
const ū = find_zero(x -> (x - 1.0)/(x + 1.0) - exp(-x/nu), 1)

# f_exact(x) = 2.0 / (1.0 + exp((x - 1.0)/nu)) - 1.0
function f_exact(x::Real)
    ū * (2/(1 + exp(ū*(x-1)/nu)) - 1)
end

function fx_exact(x::Real)
    ForwardDiff.derivative(f_exact, x)
end

const NK = 4                    # element DOF
const nodes, weights = gausslobatto(NK)
struct Element{T<:Real}
    node::Vector{T}
    data::Vector{T}
end
Element(node::Vector{T}) where {T} = Element(node, zeros(T, NK))

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

const D = h(nodes)

function element_loss(el::Element)
    u = el.data
    j = jacobian(el)
    u1 = D * u * j
    u2 = D * u1 * j
    # x = point_from_local_to_global(el, nodes)
    r =  @. u * u1 - nu * u2
    sum(r.^2 .* weights) * j
end

function loss(node_values, grid_points)
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
    loss1 = (element_loss ∘ get_element).(range(1, NE, step=1)) |> sum
    sum(nedge -> begin
        el1 = get_element(nedge)
        el2 = get_element(nedge+1)
        ((D * el1.data)[end] * jacobian(el1) - (D * el2.data)[1] * jacobian(el2))^2
        end, range(1, NE-1, step=1)) + loss1
end

function train(grid::Array)
    NE = size(grid, 1) - 1
    NN = 3NE + 1
    node_values = zeros(NN)
    node_values[1] = 1.0

    opt_f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
    prob = OptimizationProblem(opt_f, node_values, grid)
    # sol = solve(prob, ADAM(), maxiters=100)
    sol = solve(prob, BFGS())
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
    annotate!(p, 0.5, 0.01, "L2 Error = $(@sprintf("%.1e", err1))")
    #annotate!(p, 0.5, 0.03, "Number of Nodes = $(@sprintf("%2i", size(grid, 1)))")
    title!(p, "Function Value on Chebyshev Grid")

    # q = plot(xgrid, fx_exact.(xgrid), label="analytical")
    # scatter!(q, grid, fx_simu, label="simulated")
    # title!(q, "Derivative Value")
    # err2 = mean(abs2, fx_exact.(grid) - fx_simu)
    # annotate!(q, 0.5, -30, "L2 Error = $(@sprintf("%.1e", err2))")

    # plot(p, q, layout=(1, 2), size=(1000, 300), legend=:bottomleft)
    plot!(p, legend=:bottomleft)
end

function show_results(result)
    show_results(result[1], result[2])
end
#
#
# SEM works now, but derivative given by penalty, there are two things to do,
# first is try penalty on 2nd order derivative also, and second is try constrain
# instead.
