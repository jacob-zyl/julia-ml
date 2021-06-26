module FEMUtils

using LinearAlgebra, StaticArrays, Statistics

export Element, area, jacobian, jacobianr
export point_from_global_to_local, point_from_local_to_global
export point_interpolate, point_derivative, point_derivative2
export H, h, hh

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
    k = 2.0 / (x[2] - x[1])
    b = (x[1] + x[2]) / (x[1] - x[2])
    @. k * p + b
end

function point_from_local_to_global(e::Element, p)
    x = e.node
    k = (x[2] - x[1]) * 0.5
    b = (x[2] + x[1]) * 0.5
    @. k * p + b
end

H1(x) = @. 0.25 * (1.0 - x)^2 * (2.0 + x)
H2(x) = @. 0.25 * (1.0 - x)^2 * (x + 1.0)
H3(x) = @. 0.25 * (1.0 + x)^2 * (2.0 - x)
H4(x) = @. 0.25 * (1.0 + x)^2 * (x - 1.0)
h1(x) = @. 0.75 * (x^2 - 1.0)
h2(x) = @. 0.25 * (3.0x^2 - 2.0x - 1.0)
h3(x) = @. -0.75 * (x^2 - 1.0)
h4(x) = @. 0.25 * (3.0x^2 + 2.0x - 1.0)
hh1(x) = @. 1.5x
hh2(x) = @. 1.5x - 0.5
hh3(x) = @. -1.5x
hh4(x) = @. 1.5x + 0.5

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

end
