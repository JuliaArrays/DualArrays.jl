##
# Solve pendulum ODE:

# x'' + sin(x) = 0

# via discretisation and Newton's method.

# We observe that DualArrays.jl is able to solve this ODE
# Accurately and in O(n) time.
##

using LinearAlgebra, ForwardDiff, Plots, DualArrays, FillArrays, BenchmarkTools, BandedMatrices, Zygote

#Boundary Conditions
a = 0.1
b = 0.0

Tmax = 5.0
ts = 0.01

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

#LHS of ode
function f(x)
    n = length(x)
    D = Tridiagonal([ones(Float64, n) / ts ; 0.0], [1.0; -2ones(Float64, n) / ts; 1.0], [0.0; ones(Float64, n) / ts])
    (D * [a; x; b])[2:end-1] + sin.(x)
end

#Newton's method using ForwardDiff.jl
function newton_method_forwarddiff(f, x0, n)
    x = x0
    for i = 1:n
        ∇f = ForwardDiff.jacobian(f, x)
        x = x - ∇f \ f(x)
    end
    x
end

#Newton's method using DualArrays.jl
function newton_method_dualvector(f, x0, n)
    x = x0
    for i = 1:n
        ∇f = DualArrays.jacobian(f, x, BandedMatrix)
        x = x - ∇f \ f(x)
    end
    x
end

#Newton's method using Zygote.jl
function newton_method_zygote(f, x0, n)
    x = x0
    for i = 1:n
        ∇f = only(Zygote.jacobian(f, x))
        x = x - ∇f \ f(x)
    end
    x
end

# Plot times for Newton's method using DualArrays, ForwardDiff, and Zygote
function plot_times(save=undef; ns = [50, 100, 200, 400, 800], iterations = 10)
    dualvector_times = Float64[]
    forwarddiff_times = Float64[]
    zygote_times = Float64[]

    for n in ns
        println("Computing solution with n = $n")
        x0 = zeros(Float64, n)
        push!(dualvector_times, @belapsed newton_method_dualvector(f, $x0, $iterations))
        push!(forwarddiff_times, @belapsed newton_method_forwarddiff(f, $x0, $iterations))
        push!(zygote_times, @belapsed newton_method_zygote(f, $x0, $iterations))
    end

    plot(
        ns,
        dualvector_times,
        ;
        label = "DualArrays",
        title = "Pendulum Newton Solve Runtime",
        xlabel = "Number of points (n)",
        ylabel = "Runtime (seconds)",
        yticks = 10 .^ collect(-4:0.25:1),
        xticks = 10 .^ collect(1.5:0.25:3),
        plot_args...,
    )
    plot!(ns, forwarddiff_times, label = "ForwardDiff", lw = 2.5, marker = :square, color = :darkorange2)
    plot!(ns, zygote_times, label = "Zygote", lw = 2.5, marker = :diamond, color = :forestgreen)
    if save !== undef
        savefig(save)
    end
end

# Plot memory allocations for Newton's method using DualArrays, ForwardDiff, and Zygote
function plot_memory(save=undef; ns = [50, 100, 200, 400, 800], iterations = 10)
    dualvector_memory = Float64[]
    forwarddiff_memory = Float64[]
    zygote_memory = Float64[]

    for n in ns
        println("Computing solution with n = $n")
        x0 = zeros(Float64, n)
        push!(dualvector_memory, @ballocated newton_method_dualvector(f, $x0, $iterations))
        push!(forwarddiff_memory, @ballocated newton_method_forwarddiff(f, $x0, $iterations))
        push!(zygote_memory, @ballocated newton_method_zygote(f, $x0, $iterations))
    end

    plot(
        ns,
        dualvector_memory,
        ;
        label = "DualArrays",
        title = "Pendulum Newton Solve Memory",
        xlabel = "Number of points (n)",
        ylabel = "Allocated bytes",
        yticks = 10 .^ collect(5:0.5:12),
        xticks = 10 .^ collect(1.5:0.25:3),
        plot_args...,
    )
    plot!(ns, forwarddiff_memory, label = "ForwardDiff", lw = 2.5, marker = :square, color = :darkorange2)
    plot!(ns, zygote_memory, label = "Zygote", lw = 2.5, marker = :diamond, color = :forestgreen)
    if save !== undef
        savefig(save)
    end
end

# Plot solution with obtainded through newtons method with DualArrays.
# Used to verify correctness.
function plot_solution(save=undef)
    n = Int(Tmax/ts) - 1
    x0 = zeros(Float64, n)
    sol = newton_method_dualvector(f, x0, 10)
    t = ts:ts:(n * ts)
    plot(t, sol, label="Pendulum Solution", xlabel="Time", ylabel="Angle")
    if save !== undef
        savefig(save)
    end
end