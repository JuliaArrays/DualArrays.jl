using DualArrays, LinearAlgebra, Plots, BenchmarkTools, BandedMatrices, ForwardDiff, Zygote

"""
Solving the Catenary Problem using DualArrays.jl
reference: https://www.chebfun.org/examples/opt/Catenary.html

GOAL: Solve the catenary problem from variational calculus using
gradient descent and DualArrays.jl

The computation of finite differences keeps the Jacobian sparse,
so we can use DualArrays.jl
"""

const plot_args = (
    xscale = :log10,
    yscale = :log10,
    lw = 2.5,
    marker = :circle,
    color = :steelblue4,
    legend = :topleft,
    framestyle = :box,
    size = (960, 600),
    left_margin = 10Plots.mm,
    bottom_margin = 10Plots.mm,
)

function L(y, h, alpha, beta)
    """
    Evaluate functional with boundary conditions (alpha, beta)
    By approximating y' using finite differences.
    We evaluate L at the midpoints of the intervals
    using centered differences for better stability.
    """
    y_ext = [alpha; y; beta]
    dy = (y_ext[2:end] - y_ext[1:end-1]) / h
    y_mid = (y_ext[1:end-1] + y_ext[2:end]) / 2
    return y_mid .* sqrt.(1 .+ dy.^2)
end

function learn_catenary_dualarrays(h = 0.1, alpha = cosh(-1), beta = cosh(1), epochs = 2000, lr = 0.01)
    n = Int(2 / h) - 1
    y = ones(n) * (alpha + beta) / 2
    for _ = 1:epochs
        jac = DualArrays.jacobian(y -> L(y, h, alpha, beta), y, BandedMatrix)
        grads = h * sum(jac, dims=1)
        y -= lr * vec(grads)
    end
    return y
end

function learn_catenary_forwarddiff(h = 0.1, alpha = cosh(-1), beta = cosh(1), epochs = 2000, lr = 0.01)
    n = Int(2 / h) - 1
    y = ones(n) * (alpha + beta) / 2
    for _ = 1:epochs
        jac = ForwardDiff.jacobian(y -> L(y, h, alpha, beta), y)
        grads = h * sum(jac, dims=1)
        y -= lr * vec(grads)
    end
    return y
end

function learn_catenary_zygote(h = 0.1, alpha = cosh(-1), beta = cosh(1), epochs = 2000, lr = 0.01)
    n = Int(2 / h) - 1
    y = ones(n) * (alpha + beta) / 2
    for _ = 1:epochs
        jac = only(Zygote.jacobian(y -> L(y, h, alpha, beta), y))
        grads = h * sum(jac, dims=1)
        y -= lr * vec(grads)
    end
    return y
end

function plot_solution(save=undef;h = 0.1, alpha = cosh(-1), beta = cosh(1), epochs = 5000, lr = 0.02)
    x = collect(-1+h:h:1-h)
    y = learn_catenary_dualarrays(h, alpha, beta, epochs, lr)
    plot(x, y, label = "Approximate solution (DualArrays)", title="Catenary Solution", legend=:topleft)
    plot!(x, cosh.(x), label = "Exact Solution (cosh(x))", ls = :dash)
    if save !== undef
        savefig(save)
    end
end

function plot_times(save=undef; hs = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125], epochs = 200, lr = 0.01)
    ns = Int.(2 ./ hs) .- 1
    dualvector_times = Float64[]
    forwarddiff_times = Float64[]
    zygote_times = Float64[]

    for (h, n) in zip(hs, ns)
        println("Computing solution with h = $h, n = $n")
        push!(dualvector_times, @belapsed learn_catenary_dualarrays($h, cosh(-1), cosh(1), $epochs, $lr))
        push!(forwarddiff_times, @belapsed learn_catenary_forwarddiff($h, cosh(-1), cosh(1), $epochs, $lr))
        push!(zygote_times, @belapsed learn_catenary_zygote($h, cosh(-1), cosh(1), $epochs, $lr))
    end

    plot(
        ns,
        dualvector_times,
        ;
        label = "DualArrays",
        title = "Catenary Gradient Descent Runtime",
        xlabel = "Number of points (n)",
        ylabel = "Runtime (seconds)",
        yticks = 10 .^ collect(-3:0.25:2),
        xticks = 10 .^ collect(1.5:0.25:3.5),
        plot_args...,
    )
    plot!(ns, forwarddiff_times, label = "ForwardDiff", lw = 2.5, marker = :square, color = :darkorange2)
    plot!(ns, zygote_times, label = "Zygote", lw = 2.5, marker = :diamond, color = :forestgreen)
    if save !== undef
        savefig(save)
    end
end

function plot_memory(save=undef; hs = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125], epochs = 200, lr = 0.01)
    ns = Int.(2 ./ hs) .- 1
    dualvector_memory = Float64[]
    forwarddiff_memory = Float64[]
    zygote_memory = Float64[]

    for (h, n) in zip(hs, ns)
        println("Computing solution with h = $h, n = $n")
        push!(dualvector_memory, @ballocated learn_catenary_dualarrays($h, cosh(-1), cosh(1), $epochs, $lr))
        push!(forwarddiff_memory, @ballocated learn_catenary_forwarddiff($h, cosh(-1), cosh(1), $epochs, $lr))
        push!(zygote_memory, @ballocated learn_catenary_zygote($h, cosh(-1), cosh(1), $epochs, $lr))
    end

    plot(
        ns,
        dualvector_memory,
        ;
        label = "DualArrays",
        title = "Catenary Gradient Descent Memory",
        xlabel = "Number of points (n)",
        ylabel = "Allocated bytes",
        yticks = 10 .^ collect(6:0.5:11),
        xticks = 10 .^ collect(1.5:0.25:3.5),
        plot_args...,
    )
    plot!(ns, forwarddiff_memory, label = "ForwardDiff", lw = 2.5, marker = :square, color = :darkorange2)
    plot!(ns, zygote_memory, label = "Zygote", lw = 2.5, marker = :diamond, color = :forestgreen)
    if save !== undef
        savefig(save)
    end
end