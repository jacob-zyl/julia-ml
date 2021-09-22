using JLD
using FastGaussQuadrature

gen() = begin
    # fem_dict = load("prob.jld")
    ng = 10
    nn = (ng + 1)^2
    ne = ng^2

    walls(NG) = begin
        upper_wall = ((NG+1)*NG+1):(NG+1)^2
        lower_wall = 1:(NG+1)
        left_wall = 1:(NG+1):(NG*(NG+1)+1)
        right_wall = (NG+1):(NG+1):(NG+1)^2
        (upper_wall, lower_wall, left_wall, right_wall)
    end
    upper_wall, lower_wall, left_wall, right_wall = walls(ng)

    side = ng^-1 * (0:ng) |> collect
    side = @. 0.5 - 0.5cos(pi*side)
    nodes = hcat(map(x->hcat(vcat.(side', x)...), side)...)
    data = zeros(4, nn)

    data[1, upper_wall] = nodes[1, upper_wall] .|> sinpi
    # data[3, upper_wall] = (nodes[1, upper_wall] .|> sinpi) .* (pi / tanh(pi))
    # data[1, lower_wall] = nodes[1, lower_wall] .|> sinpi
    # data[1, left_wall] = -nodes[2, left_wall] .|> sinpi
    # data[1, right_wall] = 1 .- nodes[2, right_wall] .|> sinpi

    elnodes = hcat(map(p -> e2nvec(p, ng), 1:ne)...)

    @save "mesh.jld" ng ne nn elnodes nodes
    @save "data.jld" data
end

e2nvec(ne, ng) = begin
    quotient = div(ne - 1, ng)
    res = ne - ng * quotient
    n1 = quotient * (ng + 1) + res
    n2 = n1 + 1
    n4 = n2 + ng
    n3 = n4 + 1
    [n1, n2, n3, n4]
end
