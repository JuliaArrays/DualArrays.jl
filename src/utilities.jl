# Miscellaneous functions for DualArrays.jl
using FillArrays, BandedMatrices, LinearAlgebra
"""
Sum all elements of a DualVector, returning a single Dual number.
"""
function Base.sum(x::DualVector)
    n = length(x.value)
    Dual(sum(x.value), vec(sum(x.jacobian; dims=1)))
end

# Helper functions for vcat operations
_jacobian(d::Dual) = permutedims(d.partials)
_jacobian(d::DualVector) = d.jacobian
_jacobian(d::DualVector, ::Int) = d.jacobian
_jacobian(x::Number, N::Int) = Zeros(typeof(x), 1, N)

_value(v::AbstractVector) = v
_value(d::DualVector) = d.value
_value(d::Dual) = d.value
_value(x::Number) = x

_size(x::Real) = 1
_size(x::DualVector) = length(x.value)

"""
Vertically concatenate Dual numbers and DualVectors.
"""
function Base.vcat(x::Union{Dual, DualVector}...)
    value = vcat((d.value for d in x)...)
    jacobian = vcat((_jacobian(d) for d in x)...)
    DualVector(value, jacobian)
end

"""
Vertically concatenate Real numbers and DualVectors.
"""

function Base.vcat(x::Union{Real, DualVector}...)
    # Avoid stack overflow
    if all(i -> i isa Real, x)
        return Base.cat(x..., dims=1)
    end
    cols = maximum((_size(i) for i in x))
    val = vcat((_value(i) for i in x)...)
    jac = vcat((_jacobian(i, cols) for i in x)...)
    DualVector(val, jac)
end

"""
Custom display method for DualVectors.
"""
show_dual_vector(io::IO, x::AbstractArray, ::Any) = print(io, x)
function show_dual_vector(io::IO, x::DualVector, i = 0)
    show_dual_vector(io, x.value, i+1)
    print(io, " + $(x.jacobian)ϵ" * repeat('\'', i))
end
Base.show(io::IO, x::DualVector) = show_dual_vector(io, x)
Base.show(io::IO, ::MIME"text/plain", x::DualVector) = Base.show(io, x)

"""
Utility function to compute the jacobian of a function `f` at point `x`.
Analogous to `ForwardDiff.jacobian`.

`id` selects the type of (sparse) identity to use and must be either `Eye` (default) or `BandedMatrix`.
"""
function jacobian(f::Function, x::AbstractVector, id=Eye)
    J = if id === Eye
        Eye(length(x))
    elseif id === BandedMatrix
        BandedMatrix(I(length(x)), (0, 0))
    else
        Matrix(I(length(x)))
    end
    d = DualVector(x, J)
    return f(d).jacobian
end
