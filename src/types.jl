# Core type definitions for DualArrays.jl

"""
    Dual{T, Partials <: AbstractVector{T}} <: Real

A dual number type that stores a value and its partials (derivatives).

# Fields
- `value::T`: The primal value
- `partials::Partials`: The partial derivatives as a vector
"""
struct Dual{T, Partials <: AbstractVector{T}} <: Real
    value::Union{T, Dual{T}}
    partials::Partials
end



"""
    DualVector{T, M <: AbstractMatrix{T}} <: AbstractVector{Dual{T}}

Represents a vector of dual numbers given by:
    
    values + jacobian * [ε₁, …, εₙ]

For now the entries just return the values when indexed.

# Fields
- `value::Vector{T}`: The primal values
- `jacobian::M`: The Jacobian matrix containing partial derivatives

# Constructor
    DualVector(value::Vector{T}, jacobian::M) where {T, M <: AbstractMatrix{T}}

Constructs a DualVector, ensuring that the vector length matches the number of rows in the Jacobian.
"""
struct DualVector{T, V <: AbstractVector{<:Union{T, Dual{T}}},M <: AbstractMatrix{T}} <: AbstractVector{Dual{T}}
    value::V
    jacobian::M

    function DualVector(value::V, jacobian::M) where {T, V <: AbstractVector{<:Union{T, Dual{T}}}, M <: AbstractMatrix{T}}
        if size(jacobian, 1) != length(value)
            x, y = length(value), size(jacobian, 1)
            throw(ArgumentError("vector length must match number of rows in jacobian.\n" *
                               "vector length: $x\n" *
                               "no. of jacobian rows: $y"))
        end
        new{T,V, M}(value, jacobian)
    end
end

promote_type_dual(T, S) = promote_type(T, S)

promote_type_dual(::Type{<:Dual{T}}, S) where {T} = promote_type(T, S)
promote_type_dual(S, ::Type{<:Dual{T}}) where {T} = promote_type(T, S)

broadcasted(::Type{T}, d::DualVector) where {T} = DualVector(T.(d.value), T.(d.jacobian))

"""
Constructor that forces type compatibility
"""
function DualVector(value::AbstractVector{T}, jacobian::AbstractMatrix{S}) where {T, S}
    U = promote_type_dual(T, S)
    DualVector(U.(value), U.(jacobian))
end

# Basic equality for Dual numbers
==(a::Dual, b::Dual) = a.value == b.value && a.partials == b.partials
isapprox(a::Dual, b::Dual) = isapprox(a.value, b.value) && isapprox(a.partials, b.partials)