using GalacticOptim, Optim
rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0,100.0]

prob = OptimizationProblem(rosenbrock,x0,p)
sol = solve(prob, NelderMead())

f = OptimizationFunction(rosenbrock, GalacticOptim.AutoZygote())
prob2 = OptimizationProblem(f, x0, p)
@time sol2 = solve(prob2, BFGS())
