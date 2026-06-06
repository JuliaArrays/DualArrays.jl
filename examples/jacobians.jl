###
# Illustration of computing times for jacobians
# using DualArrays and ForwardDiff.
#
# We observe that for vector-valued functions with a sparse jacobian,
# performance is significantly better using DualArrays.jl
# i.e if the n x n jacobian has O(n) nonzero entries, then
# DualArrays.jl computes it in O(n) time.
#
###

using DualArrays, ForwardDiff, Plots, BenchmarkTools, BandedMatrices

f(x) = sin.(x)

function plot_times()
    dualarray_times = Float64[]
    forwarddiff_times = Float64[]
    ns = [10, 20, 50, 100, 200, 500, 1000]

    for n in ns
        println("Computing jacobian for n = $n")
        push!(dualarray_times, @belapsed DualArrays.jacobian(f, rand($n), BandedMatrix))
        push!(forwarddiff_times, @belapsed ForwardDiff.jacobian(f, rand($n)))
    end

    plot(ns, dualarray_times, label="DualArrays")
    plot!(ns, forwarddiff_times, label="ForwardDiff")
end