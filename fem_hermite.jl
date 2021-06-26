# Time-stamp: <2021-05-31 15:41:52 jacob>
using Plots
pyplot()

H1(x) = @. 0.25f0 * (1.0f0 - x)^2 * (2.0f0 + x)
H2(x) = @. 0.25f0 * (1.0f0 - x)^2 * (x + 1.0f0)
H3(x) = @. 0.25f0 * (1.0f0 + x)^2 * (2.0f0 - x)
H4(x) = @. 0.25f0 * (1.0f0 + x)^2 * (x - 1.0f0)

H(x) = [H1(x) H2(x) H3(x) H4(x)]

h1(x) = @. 0.75f0 * (x^2 - 1.0f0)
h2(x) = @. 0.25f0 * (3.0f0x^2 - 2.0f0x - 1.0f0)
h3(x) = @. -0.75f0 * (x^2 - 1.0f0)
h4(x) = @. 0.25f0 * (3.0f0x^2 + 2.0f0x - 1.0f0)
h(x) = [h1(x) h2(x) h3(x) h4(x)]

hh1(x) = @. 1.5f0x
hh2(x) = @. 0.5f0(-1.0f0 + 3.0f0x)
hh3(x) = @. -1.5f0x
hh4(x) = @. 0.5f0(1.0f0 + 3.0f0x)
hh(x) = [hh1(x) hh2(x) hh3(x) hh4(x)]

function point_from_global_to_local(x, p) # code verified
    k = 2.0f0 / (x[2] - x[1])
    b = (x[1] + x[2]) / (x[1] - x[2])
    @. k * p + b
end

function point_from_local_to_global(x, p) # code verified
    k = (x[2] - x[1]) * 0.5f0
    b = (x[2] + x[1]) * 0.5f0
    @. k * p + b
end

function element_loss(u, x)
    ## two points
    # p = [0.5773503f0, -0.5773503f0]
    # w = [1.0f0, 1.0f0]

    # four points
    p = [-0.8611363f0, -0.3399810f0, 0.3399810f0, 0.8611363f0]
    w = [0.3478548f0, 0.6521452f0, 0.6521452f0, 0.3478548f0]
    sum(point_residual(u, x, p) .* w) * 0.5f0 * (x[2] - x[1])
end

function point_interpolate_local(u, p)
    H(p) * u
end

function point_interpolate_derivative_local(u, p)
    k = (x[2] - x[1]) * 0.5f0
    rk = 1.0f0 / k
    h(p) * u * rk
end

function point_interpolate_derivative2_local(u, x, p)
    k = (x[2] - x[1]) * 0.5f0
    rk = 1.0f0 / k
    hh(p) * u * rk^2
end

function point_interpolate_global(u, x, q)
    p = point_from_global_to_local(x, q)
    point_interpolate_local(p)
end

function point_residual(u, x, p)
    k = (x[2] - x[1]) * 0.5f0
    b = (x[2] + x[1]) * 0.5f0
    rk = 1.0f0 / k^2
    a = point_interpolate_derivative2_local(u, x, p)
    b = point_interpolate_local(u, p)
    c = point_from_local_to_global(x, p)
    @. (a + b + c)^2
end

function train(N::Integer)
    train(range(0.0f0, 1.0f0, length=N))
end

function train(grid)
    # grid = [0.0f0, 0.2f0, 0.3f0, 0.4f0, 0.6f0, 0.8f0, 0.9f0, 1.0f0]
    NN = length(grid)
    NE = NN - 1
    NK = 2
    grid_values = zeros(Float32, 2, NN)

    function loss(grid_values, p)

        function get_value(e)
            if e == 1
                u1 = @views [0.0f0; grid_values[2, 1]]
                u2 = @view grid_values[:, 2]
            elseif e == NE
                u1 = @view grid_values[:, NE]
                u2 = @views [0.0f0; grid_values[2, NE+1]]
            else
                u1 = @view grid_values[:, e]
                u2 = @view grid_values[:, e+1]
            end
            [u1; u2]
        end

        function get_node(e)
            @view grid[e:e+1]
        end

        loss = 0.0f0

        for e in range(1, NE, step=1)
            loss = loss + element_loss(get_value(e), get_node(e))
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

    p = plot(xgrid, sin.(xgrid)/sin(1) .- xgrid, label="analytical")
    scatter!(p, grid, sol.minimizer[1, :], label="simulated")
    title!(p, "Function Value")


    # q = plot(xgrid, cos.(xgrid)/sin(1) .- 1, label="analytical")
    # scatter!(q, grid, sol.minimizer[2, :], label="simulated")
    # title!(q, "Derivative Value")

    # plot(p, q, layout=(1, 2))
    p
end

function show_results(result)
    show_results(result[1], result[2])
end
