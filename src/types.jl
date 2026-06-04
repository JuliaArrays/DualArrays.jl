# Core type definitions for DualArrays.jl

"""
    ArrayOperator{N, M, T, L}

This type represents a linear map from an M-array to an N-array, with a N+M=L-dimensional
array as its underlying data. This can be thought of analogously to an L-tensor equipped
with a contraction pattern characterised by (N, M). Specifically, the tensor has dimensions
a₁ x a₂ x ... x a_N x b₁ x b₂ x ... x b_M and maps an M-array of shape (b₁, b₂, ..., b_M)
to an N-array of shape (a₁, a₂, ..., a_N) by contracting over the last M indices.

We have:
-L is the dimensionality of the tensor
-T is the element type of the tensor
-N is the dimensionality of the input array/number of lower indices
-M is the dimensionality of the output array/number of upper indices

We enforce L = N + M by inferring M in the constructor.

In the context of DualArrays.jl, a DualArray can be thought of as

a + Jϵ 

Where a is an N-array of real numbers, J is an N+M=tensor and ϵ is an M-array of dual parts.
In the simplest case, where N = 0, we have a Dual number with dual parts arranged in an M-array.
"""
struct ArrayOperator{N, M, T, L, A <: AbstractArray{T, L}}
    data::A
end

# Constructor to wrap an array with a tensor, given a contraction rule represented by N
function ArrayOperator{N}(data::AbstractArray{T, L}) where {L, T, N}
    ArrayOperator{N, L - N, T, L, typeof(data)}(data)
end

# Helper convert function
elconvert(::Type{T}, t::ArrayOperator{N, M, S, L, A}) where {T, N, M, S, L, A} = ArrayOperator{N, M, T, L, typeof(elconvert(T, t.data))}(elconvert(T, t.data))

# Basic array interface
for op in (:size, :axes, :iterate)
    @eval begin
        ($op)(t::ArrayOperator) = ($op)(t.data)
        ($op)(t::ArrayOperator, i...) = ($op)(t.data, i...)
    end
end

# Since ArrayOperator is not an AbstractArray we define these manually
eltype(t::ArrayOperator) = eltype(t.data)
eltype(::Type{<:ArrayOperator{N, M, T}}) where {N,M,T} = T

Base.Broadcast.broadcastable(t::ArrayOperator) = t

sum(t::ArrayOperator; kwargs...) = sum(t.data; kwargs...)

# Equality is only defined for two ArrayOperators of the same (N, M).
==(a::ArrayOperator{N, M}, b::ArrayOperator{N, M}) where {N, M} = a.data == b.data
isapprox(a::ArrayOperator{N, M}, b::ArrayOperator{N, M}; kwargs...) where {N, M} = isapprox(a.data, b.data; kwargs...)

"""
We want broadcasting on ArrayOperators to behave
as it would on the underlying array, but preserving the
ArrayOperator{N, M, T, L} type with the correct type parameters.

T and L are preserved or inferred from type promotion.

When the highest order is an ArrayOperator, N and M of that
ArrayOperator are preserved. We do not support binary broadcasts
for cases where:

1. The highest order is not an ArrayOperator
2. We have an (N, M) ArrayOperator and an (N2, M2) ArrayOperator
    with N + M = N2 + M2 but (N, M) != (N2, M2).

As inferring N and M of the resulting ArrayOperator is ambiguous.

We define a custom broadcast style.

We inherit from the L-array broadcast style and require output dimension N
as extra information.
"""
struct ArrayOperatorBroadcastStyle{L, N} <: Broadcast.AbstractArrayStyle{L} end

Base.BroadcastStyle(::Type{<:ArrayOperator{N, <:Any, <:Any, L}}) where {L, N} = ArrayOperatorBroadcastStyle{L, N}()
function Base.BroadcastStyle(s::ArrayOperatorBroadcastStyle{L, N}, ::Broadcast.DefaultArrayStyle{M}) where {L, N, M}
    L >= M ? s : throw(ArgumentError("Array has higher dimensionality than ArrayOperator"))
end
Base.BroadcastStyle(s::Broadcast.AbstractArrayStyle, t::ArrayOperatorBroadcastStyle) = Base.BroadcastStyle(t, s)
function Base.BroadcastStyle(s1::ArrayOperatorBroadcastStyle{L1, N1}, s2::ArrayOperatorBroadcastStyle{L2, N2}) where {L1, N1, L2, N2}
    if L1 > L2
        s1
    elseif L2 > L1
        s2
    else
        throw(ArgumentError("Ambiguous output dimension for resulting ArrayOperator"))
    end
end

# Helper functions to help define broadcasting/arithmetic with ArrayOperators.
# By converting a broadcast involving ArrayOperators into a broadcast
# involving the underlying arrays.
_wrap_dual_matrix(x) = x
_unwrap_arg(t::ArrayOperator) = t.data
_unwrap_arg(bc::Broadcast.Broadcasted{<:ArrayOperatorBroadcastStyle}) = _unwrap_arg(Broadcast.materialize(bc))
_unwrap_arg(bc::Broadcast.Broadcasted) = Broadcast.materialize(bc)
_unwrap_arg(x) = x

# copy ensures that arithmetic involving a Tensor returns a Tensor
function Base.copy(bc::Broadcast.Broadcasted{ArrayOperatorBroadcastStyle{L, N}}) where {L, N}
    # We create a Broadcasted of the underlying arrays and create a Tensor containing
    # the evaluated broadcast. We check if Base.broadcasted is a Broadcasted
    # or is overriden such as with DualArrays
    ArrayOperator{N}(_wrap_dual_matrix(Broadcast.materialize(Broadcast.broadcasted(bc.f, map(_unwrap_arg, bc.args)...))))
end

# copyto adds support for .=
function Base.copyto!(dest::ArrayOperator, bc::Broadcast.Broadcasted{ArrayOperatorBroadcastStyle{L, N}}) where {L, N}
    # As above
    copyto!(dest.data, Broadcast.broadcasted(bc.f, map(_unwrap_arg, bc.args)...))
    dest
end

"""
    Dual{T, Partials <: AbstractArray{T}} <: Real

A dual number type that stores a value and its partials (derivatives).

# Fields
- `value::T`: The primal value
- `partials::Partials`: The partial derivatives stored as an array

# Constructors
    Dual(value::T, partials::AbstractArray{S}) where {S, T}

Lets us construct a Dual number with a value and an array of partials.
The partials are currently stored as an array due to a technicality in
the way that Julia differentiates scalars and 0-arrays, meaning that
having the partials as an ArrayOperator{0} would be incorrect.

    Dual(value::T, partials::ArrayOperator{0, M, S, L}) where {L, S, M, T}
    Dual(value::T, partials::ArrayOperator{N, 0, S, L}) where {L, S, N, T}

We nevertheless allow construction of a Dual using an ArrayOperator to ensure interoperability.
"""
struct Dual{T, Partials <: AbstractArray{T}} <: Real
    value::T
    # represents an ArrayOperator from an array to a scalar.
    # This is the transpose of the data stored by an ArrayOperator, so that
    # in the simple vector case we only store a vector (not a row-vector).
    partials::Partials
end

function Dual(value::T, partials::AbstractArray{S}) where {S, T}
    T2 = promote_type(T, S)
    Dual(convert(T2, value), elconvert(T2, partials))
end

function Dual(value::T, partials::ArrayOperator{0, M, S, L}) where {L, S, M, T}
    T2 = promote_type(T, S)
    Dual(convert(T2, value), elconvert(T2, partials).data)
end

# Lets us declare duals with a column vector as well as a row vector.
Dual(value::T, partials::ArrayOperator{N, 0, S, L}) where {L, S, N, T} = Dual(value, ArrayOperator{0}(partials.data))



"""
    DualArray{T, N, A <: AbstractArray{T,N}} <: AbstractArray{Dual{T}, N}

Represents a vector of dual numbers given by:
    
    values + jacobian * ϵ

Where value is an N-array of the primals, ϵ is an
M-array of dual parts and the jacobian is an N+M array of coefficients
such that right multiplication with ϵ gives a Dual N-array.

# Fields
- `value::AbstractArray{T,N}`: The primal values
- `jacobian::ArrayOperator{N, M, T, L}`: The Jacobian tensor containing partial derivatives

# Constructors
    DualArray(value::AbstractArray, jacobian::ArrayOperator)

Creates a DualArray with the given value and Jacobian, ensuring that the Jacobian
has a suitable shape and output dimensions (i.e that the jacobian when multiplying ϵ
returns an N-array of the correct shape).

    DualArray(value::AbstractArray{S, M}, jacobian::AbstractArray{T, N}) where {S, T, N, M}

We can also, given the jacobian as an AbstractArray, can infer a suitable input and output order
from the value array.
"""
struct DualArray{T, N, A <: AbstractArray{T,N}, J <: (ArrayOperator{N, M, T, L} where {L, M})} <: AbstractArray{Dual{T}, N}
    value::A
    jacobian::J

    function DualArray{T,N,A,J}(value::A, jacobian::J) where {T, N, A <: AbstractArray{T,N}, J <: (ArrayOperator{N, M, T, L} where {L, M})}
        if size(value) != ntuple(i -> size(jacobian, i), N)
            throw(ArgumentError("Length of value vector must match number of rows in Jacobian."))
        end
        new{T,N, A, J}(value, jacobian)
    end
end

DualArray{T,N}(value::A, jacobian::J) where {T, N, A <: AbstractArray{T,N}, J <: (ArrayOperator{N, M, T, L} where {L, M})} =
    DualArray{T,N,A,J}(value, jacobian)

DualArray{T,N}(value, jacobian) where {T, N} = DualArray{T,N}(elconvert(T, value), elconvert(T, jacobian))


# Constructor that forces type compatibility
function DualArray(value::AbstractArray, jacobian::ArrayOperator)
    T = promote_type(eltype(value), eltype(jacobian))
    N = ndims(value)
    DualArray{T,N}(value, jacobian)
end

# Helper function to define DualArrays with AbstractArray jacobians
function DualArray(value::AbstractArray{S, M}, jacobian::AbstractArray{T, N}) where {S, T, N, M}
    DualArray(value, ArrayOperator{M}(jacobian))
end

"""
    DualVector{T}

It is often convenient to work with this alias for a Dual 1-array.
"""
const DualVector = DualArray{T, 1} where {T}
"""
    DualVector{T}

It is often convenient to work with this alias for a Dual 2-array.
"""
const DualMatrix = DualArray{T, 2} where {T}

function DualVector(value::AbstractArray, jacobian::ArrayOperator)
    T = promote_type(eltype(value), eltype(jacobian))
    DualVector{T}(value, jacobian)
end

function DualMatrix(value::AbstractArray, jacobian::ArrayOperator)
    T = promote_type(eltype(value), eltype(jacobian))
    DualMatrix{T}(value, jacobian)
end



function DualVector(value::AbstractVector{S}, jacobian::AbstractArray{T, N}) where {S, T, N}
    DualVector(value, ArrayOperator{1}(jacobian))
end

function DualMatrix(value::AbstractMatrix{S}, jacobian::AbstractArray{T, N}) where {S, T, N}
    DualMatrix(value, ArrayOperator{2}(jacobian))
end

# If we have a matrix of dual numbers, we construct it into a DualMatrix
# This helps (for now) resolve issues surrounding broadcasting with ArrayOperator
# and DualMatrix nested (in 2nd order autodiff). A proper fix involves
# redesigning the broadcasting of dual arrays.
function _wrap_dual_matrix(x::AbstractMatrix{<:Dual})
    npartials = length(first(x).partials)
    partial_columns = reduce(hcat, vec(getfield.(x, :partials)))
    partial_tensor = permutedims(reshape(partial_columns, npartials, size(x)...), (2, 3, 1))
    DualMatrix(getfield.(x, :value), partial_tensor)
end

elconvert(::Type{Dual{T}}, a::DualVector) where {T} = DualVector(elconvert(T, a.value), elconvert(T, a.jacobian))
elconvert(::Type{Dual{T}}, a::DualMatrix) where {T} = DualMatrix(elconvert(T, a.value), elconvert(T, a.jacobian))

# Basic equality for Dual numbers
==(a::Dual, b::Dual) = a.value == b.value && a.partials == b.partials
isapprox(a::Dual, b::Dual) = isapprox(a.value, b.value) && isapprox(a.partials, b.partials)

# Type promotion on Dual
Base.promote_rule(::Type{Dual{T1}}, ::Type{Dual{T2}}) where {T1, T2} = Dual{promote_type(T1, T2)}