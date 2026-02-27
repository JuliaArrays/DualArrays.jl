# Arithmetic operations for DualArrays.jl

"""
Addition/Subtraction of DualVectors.
"""

for op in (:+, :-)
    @eval $op(x::DualVector, y::DualVector) = DualVector($op(x.value, y.value), $op(x.jacobian, y.jacobian))
    @eval $op(x::DualVector, y::AbstractVector) = DualVector($op(x.value, y), x.jacobian)
    @eval $op(x::AbstractVector, y::DualVector) = DualVector($op(x, y.value), y.jacobian)
end

"""
Matrix multiplication with a DualVector.
"""
*(x::AbstractMatrix, y::DualVector) = DualVector(x * y.value, x * y.jacobian)

"""
this section attempts to define broadcasting rules on DualVectors for functions
that either:

- take a single real argument (function applied to each element)
- are binary operations (binary operation applied to scalar and each element)

We use DiffRules.jl for symbolic derivatives and define our overloads accordingly.
"""

function diff_fn(f, n)
    """
    Helper function: Given an n-ary function f, return its partial derivatives as a tuple of function.
    """
    syms = ntuple(_ -> gensym(), n)
    d = DiffRules.diffrule(:Base, f, syms...)
    d = d isa Tuple ? d : (d,)
    makepartials(dx, syms) = eval(Expr(:->, Expr(:tuple, syms...), dx))
    return map(dx -> makepartials(dx, syms), d)
end

for (_, f, n) in DiffRules.diffrules(filter_modules=(:Base,))
    partials = diff_fn(f, n)
    if n == 1
        p = partials[1]
        @eval function broadcasted(::typeof($f), x::DualVector)
            val = $f.(x.value)
            jac = $p.(x.value) .* x.jacobian
            return DualVector(val, jac)
        end
        # Must have Base.$f in order not to import everything
        @eval Base.$f(x::Dual) = Dual($f(x.value), $p(x.value) * x.partials)
    elseif n == 2
        p1, p2 = partials
        @eval function broadcasted(::typeof($f), x::DualVector, y::Real)
            val = $f.(x.value, y)
            jac = $p1.(x.value, y) .* x.jacobian
            return DualVector(val, jac)
        end
        @eval function broadcasted(::typeof($f), x::Real, y::DualVector)
            val = $f.(x, y.value)
            jac = $p2.(x, y.value) .* y.jacobian
            return DualVector(val, jac)
        end
        @eval function broadcasted(::typeof($f), x::DualVector, y::Dual)
            val = $f.(x.value, y.value)
            # Product rule
            jac = $p1.(x.value, y.value) .* x.jacobian .+ $p2.(x.value, y.value) .* transpose(y.partials)
            return DualVector(val, jac)
        end
        @eval function broadcasted(::typeof($f), x::Dual, y::DualVector)
            val = $f.(x.value, y.value)
            # Product rule
            jac = $p1.(x.value, y.value) .* transpose(x.partials) .+ $p2.(x.value, y.value) .* y.jacobian
            return DualVector(val, jac)
        end
        @eval function broadcasted(::typeof($f), x::DualVector, y::DualVector)
            val = $f.(x.value, y.value)
            # Product rule
            jac = $p1.(x.value, y.value) .* x.jacobian .+ $p2.(x.value, y.value) .* y.jacobian
            return DualVector(val, jac)
        end
        @eval function broadcasted(::typeof($f), x::Dual, y::AbstractVector)
            val = $f.(x.value, y)
            jac = $p1.(x.value, y) .* transpose(x.partials)
            return DualVector(val, jac)
        end
        @eval function broadcasted(::typeof($f), x::AbstractVector, y::Dual)
            val = $f.(x, y.value)
            jac = $p2.(x, y.value) .* transpose(y.partials)
            return DualVector(val, jac)
        end
        # Must have Base.$f in order not to import everything
        @eval Base.$f(x::Dual, y::Dual) = Dual($f(x.value, y.value), $p1(x.value, y.value) * x.partials + $p2(x.value, y.value) * y.partials)
        @eval Base.$f(x::Dual, y::Real) = Dual($f(x.value, y), $p1(x.value, y) * x.partials)
        @eval Base.$f(x::Real, y::Dual) = Dual($f(x, y.value), $p2(x, y.value) * y.partials)
    end
end

# Special cases: integer powers
Base.:^(x::Dual, y::Integer) = Dual(x.value ^ y, y * x.value^(y - 1) * x.partials)
Base.broadcasted(::typeof(^), x::DualVector, y::Integer) = DualVector(x.value .^ y, y * x.value .^ (y - 1) .* x.jacobian)
Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::DualVector, ::Val{y}) where y = DualVector(x.value .^ y, y * x.value .^ (y - 1) .* x.jacobian)
Base.literal_pow(::typeof(^), x::Dual, ::Val{y}) where y = Dual(x.value ^ y, y * x.value^(y - 1) * x.partials)

# inner product
LinearAlgebra.dot(x::DualVector, y::DualVector) = Dual(dot(x.value, y.value), transpose(x.jacobian) * y.value + transpose(y.jacobian) * x.value)
LinearAlgebra.dot(x::DualVector, y::AbstractVector) = Dual(dot(x.value, y), transpose(x.jacobian) * y)
LinearAlgebra.dot(x::AbstractVector, y::DualVector) = Dual(dot(x, y.value), transpose(y.jacobian) * x)