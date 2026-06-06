using LinearAlgebra, DualArrays, Plots

# This example follows the following:
# https://www.dolfin-adjoint.org/en/tube_shape_derivative_demo/documentation/poisson-mother/poisson-mother.html
#
# We consider the Poisson equation with Dirichlet boundary conditions.
# Let Ω be a bounded domain in ℝⁿ. For some f: Ω → ℝ and κ ∈ ℝ we want to solve for u: Ω → ℝ such that
# -κ Δ u = f in Ω
# u = 0 on ∂Ω
# a ≤ f ≤ b in Ω
# 
# Provided f ∈ L²(Ω), there exists a unique solution u ∈ H₀¹(Ω) to the boundary value problem.
# A problem arises that given a desired solution d, we want to find f such that the solution u
# is approximately d. An intuitive way of asking this is: "given that we want a certain temperature
# field in a room, how do we position heat sources/sinks?"
#
# A way of doing this is to define an objective to be minimised given by:
#
# (1/2) ∫_Ω (u - d)² dx + (α/2) ∫_Ω f² dx
#
# Where the first term is a measurement of how far u is from d, and the second term is a Tikhonov regularisation
# with parameter α. This ensures that the problem is well-posed.
#
# We can use the adjoint method to find the gradient vector of this objective. This is so that later, we can use the Hessian to solve
# via newton's method. The Lagrangian is given by:
#
# L(u, f, p) = (1/2) ∫_Ω (u - d)² dx + (α/2) ∫_Ω f² dx + ∫_Ω p (-κ Δ u - f) dx
#
# We will use the fact that setting the derivative of the Lagrangian w.r.t u to zero gives us the adjoint equation. We use this to 
# solve for p and then obtain the gradient of the objective by differentiating the Lagrangian w.r.t f. We have:
#
# Lᵤ = (1/2) ∫_Ω (u - d)∂u dx + ∫_Ω p (-κ Δ ∂ u) dx
# L_f = ∫_Ω α f - p ∂ f dx
#
# And so we deduce that the adjoint equation is given by:
#
# -κ Δ p = u - d
#
# and the gradient of the objective is given by:
#
# ∇J(f) = α f - p
#
# We have all we need to solve this numerically by using DualArrays to do the following:
#
# - Solve for the state variable u using finite differences, i.e solve κKu = f, where K is the discretised Laplacian
# - Solve for the adjoint in the same way, -κKp = u - d
# - Analytically, we have ∇J(f) = α f - p.
#
# If we propagate f as a DualVector above, we get the f.jacobian as the Hessian of J,
# and this enables us to solve via newton's method.

"""
This calculates the gradient of the objective as detailed above.
K: the discretised Laplacian
f: the control variable
d: the desired state
alpha: the regularisation parameter
"""
function grad_objective(K, f, d, alpha)
    u = K \ f
    p = -K \ (u - d)
    return alpha * f - p
end

"""
Solve the 1D Poisson control problem by constructing the 1D discretised Laplacian
and then solving via newton's method.

f0: Initial guess
d: desired state
h: grid spacing
kappa: diffusivity constant
alpha: regularisation parameter
iters: number of iterations to run newton's method.
"""
function solve_1D(f0, d, h, kappa, alpha, iters = 30)
    fvector = copy(f0)
    n = length(f0)
    K = (kappa / h ^ 2) * Tridiagonal(-ones(n - 1), 2 * ones(n), -ones(n - 1))
    for _ = 1:iters
        grads = grad_objective(K, DualVector(fvector, I(length(fvector))), d, alpha)
        step = grads.jacobian.data \ grads.value
        fvector -= step 
    end

    return fvector
end

"""
Solve the 2D Poisson control problem by constructing the 2D discretised Laplacian
and then solving via newton's method.

f0: Initial guess
d: desired state
h: grid spacing
kappa: diffusivity constant
alpha: regularisation parameter
iters: number of iterations to run newton's method.
"""
function solve_2D(x0, d, h, kappa, alpha, iters = 30)
    fvector = copy(x0)
    n = isqrt(length(x0))
    T = Tridiagonal(-ones(n - 1), 2 * ones(n), -ones(n - 1))
    # The 2D Discretised Laplacian can be constructed as below
    # source: https://www.petercheng.me/blog/discrete-laplacian-matrix
    K = (kappa / h ^ 2) * (kron(I(n), T) + kron(T, I(n)))
    for _ = 1:iters
        grads = grad_objective(K, DualVector(fvector, I(length(fvector))), d, alpha)
        step = grads.jacobian.data \ grads.value
        fvector -= step 
    end

    return fvector
end

# Solve the 1D Poisson control on the unit interval and plot the solution.
# We compare it to a known analytical solution.
function plot_solution_1D(save=undef)

    # setup problem
    kappa = 1
    alpha = 1e-6

    T  = (0, 1)
    h = 0.02
    x = collect(T[1]:h:T[2])
    d = sin.(pi * x[2:end-1])

    # we expect this solution of f given the desired state d.
    # (this is derived using eigenfunctions of the Laplacian)
    coeff = (kappa * pi^2) / (1 + alpha * kappa^2 * pi^4)
    fexact = coeff .* sin.(pi .* x[2:end-1])

    f = solve_1D(zeros(length(d)), d, h, kappa, alpha)
    plot(x[2:end-1], f, label="Computed Control", title="1D Poisson Control Solution")
    plot!(x[2:end-1], fexact, label="Exact Control")
    if save !== undef
        savefig(save)
    end
end
    
# Solve the 2D Poisson control on the unit square and plot the solution.
# We compare it to a known analytical solution given in the dolfin-adjoint example.
function plot_solution_2D(save=undef)

    # setup problem
    kappa = 1
    alpha = 1e-6

    T = (0, 1)
    h = 0.02
    x = collect(T[1]:h:T[2])[2:end-1]
    n = length(x)

    # Create a n x n grid to calculate desired state and exact solution
    # (analogous to numpy meshgrid)
    X = repeat(reshape(x, :, 1), 1, n)
    Y = repeat(reshape(x, 1, :), n, 1)

    d = (1 / (2 * pi^2)) .* sin.(pi .* X) .* sin.(pi .* Y)

    # analytical optimal control as specified in dolfin-adjoint
    fexact = (1 / (1 + 4 * alpha * pi^4)) .* sin.(pi .* X) .* sin.(pi .* Y)

    f = solve_2D(zeros(n * n), vec(d), h, kappa, alpha)
    F = reshape(f, n, n)

    p1 = heatmap(x, x, F, title="Computed Control", aspect_ratio=1)
    p2 = heatmap(x, x, fexact, title="Exact Control", aspect_ratio=1)
    plot(p1, p2, layout=(1, 2), size=(900, 400), plot_title="2D Poisson Control Solution", plot_titlevspan=0.12)
    if save !== undef
        savefig(save)
    end
end