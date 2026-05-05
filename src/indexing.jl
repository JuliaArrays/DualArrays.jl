# Indexing operations for DualArrays.jl

using ArrayLayouts, FillArrays, LinearAlgebra, SparseArrays

sparse_getindex(a...) = layout_getindex(a...)
sparse_getindex(d::DualMatrix, a...) = getindex(d, a...)
sparse_getindex(D::Diagonal, k::Integer, ::Colon) = OneElement(D.diag[k], k, size(D, 2))
sparse_getindex(D::Diagonal, ::Colon, j::Integer) = OneElement(D.diag[j], j, size(D, 1))


# We need a system of indexing that takes two tuples
# of length N and M. We must then return a ArrayOperator whose new input and output dimensions
# are inferred from how many of the arguments in each respective tuple are integers.

# For the purposes of DualArrays.jl, we only need to index the input dimensions
#
# Example:
#
# Consider a DualVector with standard matrix Jacobian (ArrayOperator{2, T, 1, 1}):
#
# d = a + Bϵ
#
# If we index d[1], we have:
#
# d[1] = a[1] + B[1, :]ϵ
#
# Where B[1, :] indexes the input dimension of B
Idx = Union{Int, Colon, AbstractUnitRange}
count_ints(t::Tuple) = count(x -> x isa Int, t)

function getindex(t::ArrayOperator{<:Any, T, N, M}, i::NTuple{N, Idx}, j::NTuple{M, Idx}) where {T, N, M}
    # new value of M is inferred from the ArrayOperator constructor and the order of
    # indexing the underlying array.
    newN = N - count_ints(i)
    ArrayOperator{newN}(sparse_getindex(t.data, i..., j...))
end

function getindex(t::ArrayOperator{<:Any, T, N, M}, i::NTuple{N, Idx}, ::Colon) where {T, N, M}
    # new value of M is inferred from the ArrayOperator constructor and the order of
    # indexing the underlying array.
    newN = N - count_ints(i)
    ArrayOperator{newN}(sparse_getindex(t.data, i..., ntuple(_ -> Colon(), M)...))
end
# Integer indexing always returns a scalar.
getindex(t::ArrayOperator, i::Vararg{Int}) = getindex(t.data, i...)
# Indexing with only slices/ranges returns a similar tensor
getindex(t::ArrayOperator{<:Any, <:Any, N, M}, i::Vararg{Union{Colon, UnitRange}}) where {N, M} = ArrayOperator{N}(sparse_getindex(t.data, i...))

"""
Extract a single Dual number from a DualArray.
"""
function getindex(x::DualArray, args::Vararg{Int})
    Dual(x.value[args...], x.jacobian[args, :])
end

"""
Extract a single Dual number from a DualMatrix at position (y, z).
"""
function getindex(x::DualMatrix, y::Int, z::Int)
    # We collapse the epsilons into a single vector of partials.
    Dual(x.value[y, z], vec(sparse_getindex(x.jacobian, y, z, :, :)))
end

"""
Extract a row slice from a DualMatrix, returning a DualVector.

NOTE: Since a DualVector requires a 2-dimensional Jacobian, and slicing a 4-tensor returns a 3-tensor,
we implicitly 'flatten' the matrix of epsilons. This is done by the reshape.

For example, in a 2x2 case, [ϵ₁₁ ϵ₁₂;ϵ₂₁ ϵ₂₂] becomes [ϵ₁, ϵ₂, ϵ₃, ϵ₄].
"""
function getindex(x::DualMatrix, y::Int, ::Colon)
    n = size(x.value, 2)
    DualVector(x.value[y, :], reshape(sparse_getindex(x.jacobian, y, :, :, :), n, :))
end

"""
Extract a sub-DualVector from a DualVector using a range.
"""
function getindex(x::DualVector, y::UnitRange)
    newval = x.value[y]
    newjac = x.jacobian[y, :]
    DualVector(newval, newjac)
end

# DualArray array interface
for op in (:size, :axes)
    @eval $op(a::DualArray) = $op(a.value)
end