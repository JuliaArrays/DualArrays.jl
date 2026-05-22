"""
    DualArrays

A Julia package for efficient automatic differentiation using dual numbers and dual arrays.

This package provides:
- `Dual`: A dual number type for storing values and their derivatives
- `DualVector`: A vector of dual numbers represented with a Jacobian matrix
- `DualMatrix`: A matrix of dual numbers represented with a Jacobian tensor`
- `ArrayOperator`: A data structure that stores an array equipped with an output and input dimension.
    Useful for tensor computations involving higher order DualArrays
Differentiation rules are mostly provided by ChainRules.jl.
"""
module DualArrays

export DualVector, DualMatrix, Dual, jacobian, ArrayOperator

import Base: +, -, ==, getindex, size, axes, broadcasted, show, sum, vcat, convert, *, isapprox, \, eltype, transpose, permutedims

using LinearAlgebra, ArrayLayouts, FillArrays, DiffRules, TensorOperations

import FillArrays: elconvert

include("types.jl")
include("indexing.jl")
include("arithmetic.jl")
include("utilities.jl")
end

