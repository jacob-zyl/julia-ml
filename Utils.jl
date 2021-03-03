module Utils
using Printf
export show_results, get_domain, D, getxy, cliff

function show_results(ϕ, sol)
    x_test = range(0f0, 1f0, length=200)
    y_test = range(0f0, 1f0, length=200)
    func(x) = ϕ(sol.minimizer[:, 1], reshape(x, 2, 1) |> collect)[1]
    graph_f = vcat.(x_test', y_test) .|> func
    p = heatmap(x_test, y_test, graph_f, aspect_ratio=:equal)
    exact(ξ) = begin
        x, y = ξ
        sin(pi * x) * sinh(pi * y) / sinh(pi)
    end
    f_error(x, p) = func(x) - exact(x) |> abs2
    qprob = QuadratureProblem(f_error, zeros(Float32, 2), ones(Float32, 2))
    solution = solve(qprob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
    contour!(p, x_test, y_test, graph_f,
             line=(:black),
             contour_labels=true,
             levels=10,
             colorbar_entry=false,
             xlims=(0, 1),
             ylims=(0, 1))
    annotate!(p, 0.5, 0.2, "Error = $(@sprintf("%.1e", sqrt(solution.u[1])))")
    #q = contourf(x_test, y_test, (x, y) -> f([x; y], sol.minimizer)[1])
    (sqrt(solution.u[1]), p)
end

function get_domain(dim, size; show=false)
    domain = rand(Float32, dim, size)
    if show
        p = scatter(zip(domain[1, :], domain[2, :]) |> collect,
                    aspect_ratio=:equal,
                    framestyle=:box,
                    ticks=0:0.2:1,
                    xlims=(0, 1),
                    ylims=(0, 1))
        return (p, domain)
    end
    domain
end

using Zygote
D(phi, f, x) = pullback(ξ -> phi(f, ξ), x)[2](ones(Float32, 1, size(x, 2)))[1]

cliff(x; a=1.0f-3, b=1.0f5) = x + log1p(b * x) * a # This is magic (number)

getxy(a::Array{Float32, 2}) = ([1.0f0 0.0f0] * a, [0.0f0 1.0f0] * a)
end
