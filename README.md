Julia Machine Learning Trials
=============================

Only for saving, not a true repository

Notes
-----

Something to compare:

1.  Define equation differentials with `ForwardDiff.gradient`,
    `Zygote.pushforward` and `Zygote.pullback`.
2.  Apply or not `swift-sink` function to reduce and loss respectively.

Breakthroughs
-------------

  ----------- -------------------------------------------
  pinn.jl     `ForwardDiff.gradient()` works
  pinn2.jl    `Zygote.pushforward()` works
  pinn3.jl    `Optim` with FD works
  pinn4.jl    `ForwardDiff.gradient()` enhanced
  pinn5.jl    `Zygote.pullback()` enhanced
  pinn6.jl    `Flux.Optimise` with mini-batch
  pinn7.jl    `pullback` analysis
  pinn8.jl    2D problem with `pullback` and `Optim` FD
  pinn9.jl    2D problem with multi-nets
  pinn10.jl   `pushforward` and `Optim` FD
  pinn11.jl   `Optim` FD and constrains
  pinn12.jl   PERFECT
  pinn13.jl   Neumann boundary condition tested
  ----------- -------------------------------------------

The Hermite FEM
---------------

fem1.jl fem2.jl fem3.jl fem4.jl fem5.jl are very old test files.

  ------------------ ------------------------------
  fem_hermite.jl    linear ODE
  fem_hermite2.jl   `struct Element` defined
  fem_hermite3.jl   put everyting to FEMUtils.jl
  fem_hermite4.jl   nonlinear ODE
  ------------------ ------------------------------

  --------------------------- ------------------------------------------------------
  fem_hermite2d.jl           A very basic implementation
  fem_hermite2ddropgrad.jl   A truly working one
  fem_hermite2dv2.jl         Less constants in global namespace
  fem_hermite2dv3.jl         A failed trial on $f_{xy}$.
  fem_hermtie2dv4.jl         Isoparametric element implementation
  fem_hermtie2dv5.jl         Complete isoparametric element
  fem_hermite2dv6.jl         `ConjugateGradient()` used.
  fem_hermite2dv7.jl         Failed trial on `dropgrad()` position in source code
  fem_hermite2dv8.jl         Summation optimized.
  fem_hermite2dv9.jl         Summation optimized again and code beautified.
  fem_hermite2dv10.jl        Transient code deployed.
  --------------------------- ------------------------------------------------------

The Spectral Element Method
---------------------------

  --------- -----------------------------------------------------------
  sem.jl    nonlinear ODE, not working, maybe due to wrong derivative
  sem2.jl   linear ODE, not working
  sem3.jl   C1 constrain by penalty, derivative calculation corrected
  sem4.jl   C1 constrain by penalty, nonlinear ODE
  sem5.jl   C1 constrain implemented
  sem6.jl   C2 constrain implemented
  sem7.jl   half-mode constrain implemented, not working
  --------- -----------------------------------------------------------

### Benchmark test

  -------------- --------- -------- -------- ---------
  $nu = 0.005$   Hermite   SEM C1   SEM C2   Hermite
  No. Element    30        20       20       50
  Error          1.4e-4    1.8e-5   1.4e-5   1.8e-5
  Wall Time      1.842     18.000   26.650   121.841
  -------------- --------- -------- -------- ---------

  -------------- --------- -------- --------
  $nu = 0.005$   Hermite   SEM C1   SEM C2
  No. Element    15        10       10
  Error          2.0e-3    1.5e-3   1.2e-3
  Wall Time      0.503     4.179    1.615
  -------------- --------- -------- --------

Heat Equation
-------------

  ------------- -----------------------------------------------------
  heat2d.jl     The working code
  heat2dv2.jl   Failed code trying the convection form
  heat2dv3.jl   The same code as heat2d.jl
  heat2dv4.jl   Variational methods tried in unsteady heat equation
  ------------- -----------------------------------------------------

Sod Problem
-----------

  ------- ---------
  sod.jl  just a trial and error
  sod2.jl works
  sod3.jl another trial and error on operator splitting
  sod4.jl true 1st order solver
  sod5.jl the correct 2nd order solver by subcell method
  sod6.jl the incorrect 2nd order solver by enforcing spline interpolation
  ------- ----------

Convection-Diffusion Equation
-----------------------------

  ------ -----
  cd.jl  the working one
  cd2.jl operator splitting
  cd3.jl DEUtils.jl added
  cd4.jl global evalutation corrected
  cd5.jl variational form of time scheme tried (but failed)
  ------ ----------

Struggles (DONE)
---------

`GalacticOptim` now works only on FD instead of RD like `AutoZygote()`,
`AutoTracker()` or `AutoReverseDiff()`. If `ForwardDiff` is used,
`Zygote` would complain that mutating array is not supported; if
`pullback` is used, `Zygote` would complain that there is a mysterious
broadcast error.

The previous problem is solved by using `Buffer()`.
