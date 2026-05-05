"""
    DualArrays

A Julia package for efficient automatic differentiation using dual numbers and dual arrays.

This package provides:
- `Dual`: A dual number type for storing values and their derivatives
- `DualVector`: A vector of dual numbers represented with a Jacobian matrix
- `DualMatrix`: A matrix of dual numbers represented with a Jacobian tensor`
Differentiation rules are mostly provided by ChainRules.jl.
"""
module DualArrays

export DualVector, DualMatrix, Dual, jacobian

import Base: +, -, ==, getindex, size, axes, broadcasted, show, sum, vcat, convert, *, isapprox, \, eltype, transpose, permutedims

using LinearAlgebra, ArrayLayouts, FillArrays, DiffRules, TensorOperations

include("types.jl")
include("indexing.jl")
include("arithmetic.jl")
include("utilities.jl")
end

