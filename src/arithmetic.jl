# Arithmetic operations for DualArrays.jl

# Tensor transpose
transpose(t::ArrayOperator{N, M}) where {N, M} = ArrayOperator{M}(transpose(t.data))

# Addition/Subtraction of DualVectors.
for op in (:+, :-)
    @eval $op(x::DualVector, y::DualVector) = DualVector($op(x.value, y.value), $op(x.jacobian, y.jacobian))
    @eval $op(x::DualVector, y::AbstractVector) = DualVector($op(x.value, y), x.jacobian)
    @eval $op(x::AbstractVector, y::DualVector) = DualVector($op(x, y.value), y.jacobian)
end


# Matrix multiplication with a DualVector.
*(x::AbstractMatrix, y::DualVector) = DualVector(x * y.value, x * y.jacobian)

###
# this section attempts to define broadcasting rules on DualVectors for functions
# that either:
#
# - take a single real argument (function applied to each element)
# - are binary operations (binary operation applied to scalar and each element)
#
# We use DiffRules.jl for symbolic derivatives and define our overloads accordingly.
###

# Helper function: Given an n-ary function f, return its partial derivatives as a tuple of function.
function diff_fn(f, n)
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
        @eval function broadcasted(::typeof($f), x::DualVector, y::AbstractVector)
            val = $f.(x.value, y)
            jac = $p1.(x.value, y) .* x.jacobian
            return DualVector(val, jac)
        end
        @eval function broadcasted(::typeof($f), x::AbstractVector, y::DualVector)
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

# solve
\(x::AbstractMatrix, y::DualVector) = DualVector(x \ y.value, ArrayOperator{1}(x \ y.jacobian.data))



"""
Multiplication (*) for ArrayOperators generalises from matrix/vector
multiplication and uses tensor contraction provided by
TensorOperations.jl. Let an (A, B) ArrayOperator denote
an ArrayOperator with output dimension A and input dimension B
(i.e an ArrayOperator{A+B, T, A, B} for some type T). We only
allow multiplication between an (A, B) ArrayOperator and a (B, C)
ArrayOperator, with the result being an (A, C) ArrayOperator.
This generalises from:

- An inner product: a row vector * a column vector -> a scalar:

            (0, 1) * (1, 0) -> (0, 0)

- An outer product: a column vector * a row vector -> a matrix:

            (1, 0) * (0, 1) -> (1, 1)

- matrix/vector multiplication: a matrix * a column vector -> a column vector:

            (1, 1) * (1, 0) -> (1, 0)

We can treat multiplication between ArrayOperators and regular arrays as multiplication
between two ArrayOperators: given an (A, B) ArrayOperator and an L-array, we can fix it as
having input dimension B and infer C = L - B. We use this contraction rule to multiply the two.
"""
function _contract(x, y, A, B, C)
    x_idx = (ntuple(i -> i, A), ntuple(i -> i + A, B))
    y_idx = (ntuple(i -> i, B), ntuple(i -> i + B, C))
    ret_idx = (ntuple(i -> i, A), ntuple(i -> i + A, C))

    return TensorOperations.tensorcontract(x, x_idx, false, y, y_idx, false, ret_idx, 1)
end


function *(x::ArrayOperator{A, B}, y::ArrayOperator{B, C}) where {A, B, C}
    return ArrayOperator{A}(_contract(x.data, y.data, A, B, C))
end

function *(x::ArrayOperator{A, B}, y::AbstractArray{<:Any, L}) where {A, B, L}
    C = L - B
    return ArrayOperator{A}(_contract(x.data, y, A, B, C))
end

function *(x::AbstractArray{<:Any, L}, y::ArrayOperator{B, C}) where {L, B, C}
    A = L - B
    return ArrayOperator{A}(_contract(x, y.data, A, B, C))
end

# Extra required arithmetic for ArrayOperators
Base.:+(x::ArrayOperator, y::ArrayOperator) = x .+ y
Base.:-(x::ArrayOperator, y::ArrayOperator) = x .- y
Base.:-(x::ArrayOperator) = (-).(x)

Base.:*(a::Number, t::ArrayOperator{N, M, T, L}) where {N, M, T, L} =
    ArrayOperator{N, M, promote_type(typeof(a), T), L}(a .* t.data)
Base.:*(t::ArrayOperator, a::Number) = a * t


