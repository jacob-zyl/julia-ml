module DEUtils

using FastGaussQuadrature
using Zygote

const NK = 4
const NK_LESS = 2

const P, W = gausslegendre(NK)

const POINTS_2D = tuple.(P', P) |> vec

const N_POINTS = tuple.(P, 1.0ones(NK))
const S_POINTS = tuple.(P, -1.0ones(NK))
const W_POINTS = tuple.(-1.0ones(NK), P)
const E_POINTS = tuple.(1.0ones(NK), P)
    
const WEIGHTS_2D = kron(W, W)

const P_LESS, W_LESS = gausslegendre(NK_LESS)

const POINTS_2D_LESS = tuple.(P_LESS', P_LESS) |> vec

const WEIGHTS_2D_LESS = kron(W_LESS, W_LESS)

# This is basis of Hermite interpolant
h1(x) = (1.0 - x)^2 * (2.0 + x) * 0.25
h2(x) = (1.0 - x)^2 * (x + 1.0) * 0.25
h3(x) = (1.0 + x)^2 * (2.0 - x) * 0.25
h4(x) = (1.0 + x)^2 * (x - 1.0) * 0.25

hermite2d(p::Tuple{Float64, Float64}) = [
    h1(p[1])*h1(p[2]), h2(p[1])*h1(p[2]), 
    h1(p[1])*h2(p[2]), h2(p[1])*h2(p[2]),
    
    h3(p[1])*h1(p[2]), h4(p[1])*h1(p[2]), 
    h3(p[1])*h2(p[2]), h4(p[1])*h2(p[2]),
    
    h3(p[1])*h3(p[2]), h4(p[1])*h3(p[2]), 
    h3(p[1])*h4(p[2]), h4(p[1])*h4(p[2]),
    
    h1(p[1])*h3(p[2]), h2(p[1])*h3(p[2]), 
    h1(p[1])*h4(p[2]), h2(p[1])*h4(p[2])]'
hermite2d(ps::Vector{Tuple{Float64, Float64}}) = begin 
    vcat(hermite2d.(ps)...)
end

hermite2d_xx(p::Tuple{Float64, Float64}) = [
    (h1')'(p[1])*h1(p[2]), (h2')'(p[1])*h1(p[2]), 
    (h1')'(p[1])*h2(p[2]), (h2')'(p[1])*h2(p[2]),

          
    (h3')'(p[1])*h1(p[2]), (h4')'(p[1])*h1(p[2]), 
    (h3')'(p[1])*h2(p[2]), (h4')'(p[1])*h2(p[2]),
          
    (h3')'(p[1])*h3(p[2]), (h4')'(p[1])*h3(p[2]), 
    (h3')'(p[1])*h4(p[2]), (h4')'(p[1])*h4(p[2]),
          
    (h1')'(p[1])*h3(p[2]), (h2')'(p[1])*h3(p[2]), 
    (h1')'(p[1])*h4(p[2]), (h2')'(p[1])*h4(p[2])]'
hermite2d_xx(ps::Vector{Tuple{Float64, Float64}}) =  begin
    vcat(hermite2d_xx.(ps)...)
end

hermite2d_yy(p::Tuple{Float64, Float64}) = [
    h1(p[1])*(h1')'(p[2]), h2(p[1])*(h1')'(p[2]), 
    h1(p[1])*(h2')'(p[2]), h2(p[1])*(h2')'(p[2]),
          
    h3(p[1])*(h1')'(p[2]), h4(p[1])*(h1')'(p[2]), 
    h3(p[1])*(h2')'(p[2]), h4(p[1])*(h2')'(p[2]),
          
    h3(p[1])*(h3')'(p[2]), h4(p[1])*(h3')'(p[2]), 
    h3(p[1])*(h4')'(p[2]), h4(p[1])*(h4')'(p[2]),
          
    h1(p[1])*(h3')'(p[2]), h2(p[1])*(h3')'(p[2]), 
    h1(p[1])*(h4')'(p[2]), h2(p[1])*(h4')'(p[2])]'
hermite2d_yy(ps::Vector{Tuple{Float64, Float64}}) = begin
    vcat(hermite2d_yy.(ps)...)
end

hermite2d_x(p::Tuple{Float64, Float64}) = [
    (h1')(p[1])*h1(p[2]), (h2')(p[1])*h1(p[2]), 
    (h1')(p[1])*h2(p[2]), (h2')(p[1])*h2(p[2]),
          
    (h3')(p[1])*h1(p[2]), (h4')(p[1])*h1(p[2]), 
    (h3')(p[1])*h2(p[2]), (h4')(p[1])*h2(p[2]),
          
    (h3')(p[1])*h3(p[2]), (h4')(p[1])*h3(p[2]), 
    (h3')(p[1])*h4(p[2]), (h4')(p[1])*h4(p[2]),
          
    (h1')(p[1])*h3(p[2]), (h2')(p[1])*h3(p[2]), 
    (h1')(p[1])*h4(p[2]), (h2')(p[1])*h4(p[2])]'
hermite2d_x(ps::Vector{Tuple{Float64, Float64}}) = begin
    vcat(hermite2d_x.(ps)...)
end

hermite2d_y(p::Tuple{Float64, Float64}) = [
    h1(p[1])*(h1')(p[2]), h2(p[1])*(h1')(p[2]), 
    h1(p[1])*(h2')(p[2]), h2(p[1])*(h2')(p[2]),
          
    h3(p[1])*(h1')(p[2]), h4(p[1])*(h1')(p[2]), 
    h3(p[1])*(h2')(p[2]), h4(p[1])*(h2')(p[2]),
          
    h3(p[1])*(h3')(p[2]), h4(p[1])*(h3')(p[2]), 
    h3(p[1])*(h4')(p[2]), h4(p[1])*(h4')(p[2]),
          
    h1(p[1])*(h3')(p[2]), h2(p[1])*(h3')(p[2]), 
    h1(p[1])*(h4')(p[2]), h2(p[1])*(h4')(p[2])]'
hermite2d_y(ps::Vector{Tuple{Float64, Float64}}) = begin
    vcat(hermite2d_y.(ps)...)
end

hermite1d(x::Float64) = [h1(x) h2(x) h3(x) h4(x)]
hermite1d(x::Vector{Float64}) = vcat(hermite1d.(x)...)

hermite1d_derivative(x::Real) = [(h1')(x) (h2')(x) (h3')(x) (h4')(x)]
hermite1d_derivative(x::Vector) = vcat(hermite1d_derivative.(x)...)

const H_2D   = hermite2d(POINTS_2D)
const HX_2D  = hermite2d_x(POINTS_2D)
const HY_2D  = hermite2d_y(POINTS_2D)
const HXX_2D = hermite2d_xx(POINTS_2D)
const HYY_2D = hermite2d_yy(POINTS_2D)

const N_H = hermite2d(N_POINTS)
const S_H = hermite2d(S_POINTS)
const W_H  = hermite2d(W_POINTS)
const E_H = hermite2d(E_POINTS)

const N_HX = hermite2d_x(N_POINTS)
const S_HX = hermite2d_x(S_POINTS)
const W_HX = hermite2d_x.(W_POINTS)
const E_HX = hermite2d_x(E_POINTS)

const N_HY = hermite2d_y(N_POINTS)
const S_HY = hermite2d_y(S_POINTS)
const W_HY = hermite2d_y(W_POINTS)
const E_HY = hermite2d_y(E_POINTS)


const N_HXX = hermite2d_xx(N_POINTS)
const S_HXX = hermite2d_xx(S_POINTS)
const W_HXX = hermite2d_xx(W_POINTS)
const E_HXX = hermite2d_xx(E_POINTS)

const N_HYY = hermite2d_yy(N_POINTS)
const S_HYY = hermite2d_yy(S_POINTS)
const W_HYY = hermite2d_yy(W_POINTS)
const E_HYY = hermite2d_yy(E_POINTS)

const H_1D = hermite1d(P)
const HX_1D = hermite1d_derivative(P)

const H_2D_LESS = hermite2d(POINTS_2D_LESS)
const HX_2D_LESS =hermite2d_x(POINTS_2D_LESS)
const HY_2D_LESS = hermite2d_y(POINTS_2D_LESS)
const HXX_2D_LESS = hermite2d_xx(POINTS_2D_LESS)
const HYY_2D_LESS = hermite2d_yy(POINTS_2D_LESS)


const H_1D_LESS = hermite1d(P_LESS)
const HX_1D_LESS = hermite1d_derivative(P_LESS)

const WH_1D_LESS = W_LESS' * H_1D_LESS
const WH_2D_LESS = WEIGHTS_2D_LESS' * H_2D_LESS

const WH_2D  = WEIGHTS_2D' * H_2D
const WH_1D = W' * H_1D

value_on_points_1d(data, points) = hermite1d(points) * data
value_on_points_2d(data, points) = hermite2d(points) * data

value_on_points_1d(data) = H_1D * data
value_on_points_2d(data) = H_2D * data

quad_1d(data) = WH_1D ⋅ data
quad_2d(data) = WH_2D ⋅ data

end