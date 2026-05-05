# Core type definitions for DualArrays.jl

"""
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
struct ArrayOperator{L, T, N, M}
    data::AbstractArray{T, L}
end

# Constructor to wrap an array with a tensor, given a contraction rule represented by N
function ArrayOperator{N}(data::AbstractArray{T, L}) where {L, T, N}
    ArrayOperator{L, T, N, L - N}(data)
end

# Helper convert function
_convert_array(::Type{T}, a::AbstractArray) where {T} = T.(a)
_convert_array(::Type{T}, t::ArrayOperator{L, S, N, M}) where {T, L, S, N, M} = ArrayOperator{L, T, N, M}(_convert_array(T, t.data))

# Basic array interface
for op in (:size, :axes, :iterate)
    @eval begin
        ($op)(t::ArrayOperator) = ($op)(t.data)
        ($op)(t::ArrayOperator, i...) = ($op)(t.data, i...)
    end
end

# Since ArrayOperator is not an AbstractArray we define these manually
eltype(t::ArrayOperator) = eltype(t.data)

Base.Broadcast.broadcastable(t::ArrayOperator) = t

transpose(t::ArrayOperator) = transpose(t.data)
sum(t::ArrayOperator; kwargs...) = sum(t.data; kwargs...)

==(a::ArrayOperator, b::ArrayOperator) = a.data == b.data
==(a::ArrayOperator, b::AbstractArray) = a.data == b
==(a::AbstractArray, b::ArrayOperator) = a == b.data
isapprox(a::ArrayOperator, b::ArrayOperator; kwargs...) = isapprox(a.data, b.data; kwargs...)
isapprox(a::ArrayOperator, b::AbstractArray; kwargs...) = isapprox(a.data, b; kwargs...)
isapprox(a::AbstractArray, b::ArrayOperator; kwargs...) = isapprox(b, a)

# Below we define a broadcast style for ArrayOperators and override copy and copyto!
# This allows all arithmetic/broadcasting with ArrayOperators to be handled by the
# underlying logic of the array contained in the struct, while ensuring that
# all results stay as a ArrayOperator.

struct ArrayOperatorBroadcastStyle{N} <: Broadcast.AbstractArrayStyle{0} end

# N is the input dimension of the tensor being broadcasted.
# For he result of the broadcast we will choose to preserve the highest input dimension. 
Base.BroadcastStyle(::Type{<:ArrayOperator{<:Any, <:Any, N, <:Any}}) where {N} = ArrayOperatorBroadcastStyle{N}()
Base.BroadcastStyle(::ArrayOperatorBroadcastStyle{N}, ::Broadcast.DefaultArrayStyle{M}) where {N, M} = ArrayOperatorBroadcastStyle{N}()
Base.BroadcastStyle(::Broadcast.DefaultArrayStyle{M}, ::ArrayOperatorBroadcastStyle{N}) where {N, M} = ArrayOperatorBroadcastStyle{N}()
Base.BroadcastStyle(::ArrayOperatorBroadcastStyle{N}, ::ArrayOperatorBroadcastStyle{M}) where {N, M} = ArrayOperatorBroadcastStyle{max(N, M)}()

# Helper functions to help define broadcasting/arithmetic with ArrayOperators.
# By converting a broadcast involving ArrayOperators into a broadcast
# involving the underlying arrays.
_unwrap(t::ArrayOperator) = t.data
_unwrap(bc::Broadcast.Broadcasted) = Broadcast.Broadcasted(bc.f, _unwrap_args(bc.args), bc.axes)
_unwrap(x) = x
_unwrap_args(args::Tuple) = map(_unwrap, args)

# copy ensures that arithmetic involving a ArrayOperator returns a ArrayOperator
function Base.copy(bc::Broadcast.Broadcasted{ArrayOperatorBroadcastStyle{N}}) where {N}
    # We create a Broadcasted of the underlying arrays and create a ArrayOperator containing
    # the evaluated broadcast
    databroadcast = Broadcast.Broadcasted(bc.f, _unwrap_args(bc.args), bc.axes)
    ArrayOperator{N}(copy(Broadcast.flatten(databroadcast)))
end

# copyto adds support for .=
function Base.copyto!(dest::ArrayOperator, bc::Broadcast.Broadcasted{ArrayOperatorBroadcastStyle{N}}) where {N}
    # As above
    databroadcast = Broadcast.Broadcasted(bc.f, _unwrap_args(bc.args), bc.axes)
    copyto!(dest.data, Broadcast.flatten(databroadcast))
    dest
end

"""
    Dual{T, Partials <: AbstractArray{T}} <: Real

A dual number type that stores a value and its partials (derivatives).

# Fields
- `value::T`: The primal value
- `partials::Partials`: The partial derivatives stored as an array

NOTE: Partials will soon be in the ArrayOperator format. 
"""
struct Dual{T, Partials <: AbstractArray{T}} <: Real
    value::T
    partials::Partials
end

function Dual(value::T, partials::AbstractArray{S}) where {S, T}
    T2 = promote_type(T, S)
    Dual(convert(T2, value), _convert_array(T2, partials))
end

function Dual(value::T, partials::ArrayOperator{L, S, 0, M}) where {L, S, M, T}
    T2 = promote_type(T, S)
    Dual(convert(T2, value), _convert_array(T2, partials).data)
end

"""
    DualVector{T, V <: AbstractVector{T}, M <: AbstractMatrix{T}} <: AbstractVector{Dual{T}}

Represents a vector of dual numbers given by:
    
    values + jacobian * [ε₁, …, εₙ]

# Fields
- `value::Vector{T}`: The primal values
- `jacobian::M`: The Jacobian matrix containing partial derivatives

# Constructor
    DualVector(value::Vector{T}, jacobian::M) where {T, M <: AbstractMatrix{T}}

Constructs a DualVector, ensuring that the vector length matches the number of rows in the Jacobian.
"""
struct DualArray{T, N   , A <: AbstractArray{T,N},J <: (ArrayOperator{L, T, N, M} where {L, M})} <: AbstractVector{Dual{T}}
    value::A
    jacobian::J

    function DualArray(value::A, jacobian::J) where {T, N, A <: AbstractArray{T,N}, J <: (ArrayOperator{L, T, N, M} where {L, M})}
        if size(value) != ntuple(i -> size(jacobian, i), N)
            throw(ArgumentError("Length of value vector must match number of rows in Jacobian."))
        end
        new{T,N, A, J}(value, jacobian)
    end
end

"""
Constructor that forces type compatibility
"""
function DualArray(value::AbstractArray, jacobian::ArrayOperator)
    T = promote_type(eltype(value), eltype(jacobian))
    DualArray(_convert_array(T, value), _convert_array(T, jacobian))
end

# Helper function to define DualArrays with AbstractArray jacobians
function DualArray(value::AbstractArray{S, M}, jacobian::AbstractArray{T, N}) where {S, T, N, M}
    DualArray(value, ArrayOperator{M}(jacobian))
end

const DualVector = DualArray{T, 1} where {T}
const DualMatrix = DualArray{T, 2} where {T}

DualVector(value::AbstractVector, jacobian) = DualArray(value, jacobian)
DualMatrix(value::AbstractMatrix, jacobian) = DualArray(value, jacobian)
# Basic equality for Dual numbers
==(a::Dual, b::Dual) = a.value == b.value && a.partials == b.partials
isapprox(a::Dual, b::Dual) = isapprox(a.value, b.value) && isapprox(a.partials, b.partials)