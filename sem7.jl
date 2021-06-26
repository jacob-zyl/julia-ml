# The correct SEM
#

using LinearAlgebra, StaticArrays, Statistics

using GalacticOptim, Optim
using Roots, FastGaussQuadrature

using ForwardDiff

using Printf

using Plots
pyplot()

const nu = 0.01
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
    jr = jacobianr(el)
    u1 = D * u * j
    u2 = D * u1 * j
    u3 = D * u2 * j
    # x = point_from_local_to_global(el, nodes)
    r =  @. (u * u1 - nu * u2)^2 #+ (u1^2 + u * u2 - nu * u3)^2 * jr^2
    sum(r .* weights) * j
end

function element_loss(el::Element, u1)
    u = el.data
    j = jacobian(el)
    u2 = D * u1 * j
    r = @. u * u1 - nu * u2
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
    node_derivatives = zeros(size(node_values))
    elements = get_element.(range(1, NE, step=1))
    get_u1(n) = begin
        vec0 = [1.0, 1.0, 1.0, 0.5]
        vec00 = [0.0, 0.0, 0.0, 0.5]
        vec1 = [0.5, 1.0, 1.0, 0.5]
        vec2 = [0.5, 1.0, 1.0, 1.0]
        vec20 = [0.5, 0.0, 0.0, 0.0]
        u1tmp = D * elements[n].data * jacobian(elements[n])
        if n == 1
            u1_tmp_p1 = D * elements[2].data * jacobian(elements[2])
            u1 = vec0 .* u1tmp + vec00 .* u1_tmp_p1[1]
        elseif n == NE
            u1_tmp_m1 = D * elements[NE-1].data * jacobian(elements[NE-1])
            u1 = vec2 .* u1tmp + vec20 .* u1_tmp_m1[end]
        else
            u1_tmp_m1 = D * elements[n-1].data * jacobian(elements[n-1])
            u1_tmp_p1 = D * elements[n+1].data * jacobian(elements[n+1])
            u1 = vec1 .* u1tmp + vec20 .* u1_tmp_m1[end] + vec00 .* u1_tmp_p1[1]
        end
        u1
    end
    # sum([element_loss(elements[n]) for n in range(1, NE, step=1)])
    sum([element_loss(elements[n], get_u1(n)) for n in range(1, NE, step=1)])
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
