##
# Solve the p-Laplacian by minimising energy function using
# 2nd-order Newton's method
#
# The p-Laplacian PDE (https://en.wikipedia.org/wiki/P-Laplacian):
#
#   -Δ_p u  =  f   in Ω = (0, 1)
#        u  =  0   on ∂Ω
#
# Its weak solution is the minimiser of the energy functional
#
#   J(u) = (1/p) ∫₀¹ |u'(x)|ᵖ dx  -  ∫₀¹ f(x) u(x) dx
#
# We can discretise this and solve for 0 using 2nd order Newton's method.
##

using LinearAlgebra, DualArrays, Plots, BandedMatrices

function energy_function(p, f, h)
    n = length(f)
    D = BandedMatrix(0 => ones(n), -1 => -ones(n))[:, 1:end-1] / h
    return u -> (h / p) * sum(abs.(D * u) .^ p) - h * dot(f, u)
end


function newton_solve(J, u, n; n_iter = 20)
    for _ in 1:n_iter
        ∇J = J(DualVector(u, I(n))).partials
        H  = hessian(J, u)
        u  = u - H \ ∇J
    end
    return u
end

# Exact solution for f = 1 on [0,1]
exact(x, p) = begin
    q = p / (p - 1)
    ((1 / 2)^q .- abs.(x .- 1 / 2) .^ q) ./ q
end

function solve(n = 40, p = 2.0)
    h  = 1.0 / (n + 1)
    xs = range(h, step = h, length = n)
    f  = ones(n)
    J  = energy_function(p, f, h)
    u0 = 0.1 .* sin.(pi .* xs)
    u  = newton_solve(J, u0, n; n_iter = 20)
    return xs, u
end

function plot_solution(save=undef, n = 40, ps = [2.0, 2.25, 2.5, 2.75, 3.0])
    plt = plot()
    for p in ps
        xs, u = solve(; n, p)
        plot!(plt, xs, u, label = "Newton (DualArrays), p = $p", lw = 2)
        plot!(plt, xs, exact(xs, p), label = "Exact, p = $p", lw = 2, ls = :dash)
    end
    xlabel!(plt, "x")
    ylabel!(plt, "u(x)")
    title!(plt, "p-Laplacian computed vs analytical solutions")
    display(plt)
    if save !== undef
        savefig(plt, save)
    end
end
