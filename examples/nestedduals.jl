###
#
# IDEA: Can we have a DualVector inside a DualVector in order to
# achieve 'nested dual numbers' thereby achieving higher order derivatives?
# 
# BACKGROUND:
#
# Consider a number with two dual parts, ϵ and ϵ' such that ϵϵ' ≠ 0.
# We can represent such a number as follows:
#
# a + bϵ + cϵ' + dϵϵ'
#
# Consider an arbitrary function f: ℝ → ℝ given by f(x) = x^n.
# We can evaluate f at the above dual number as follows:
#
# f(a + bϵ + cϵ' + dϵϵ') =
# a^n + n a^(n-1) b ϵ + n a^(n-1) c ϵ' + (n a^(n-1) d + n(n-1) a^(n-2) b c) ϵϵ'
#
# Fix a ∈ ℝ. We observe that if we fix b = 1, c = 1, d = 0, we have that the
# coefficient of ϵϵ' is equivalent to f''(a). Thus using dual numbers like this
# Gives us a way to compute second derivatives.
#
# We now generalise to the vector case. Consider vectors of dual parts ϵ and ϵ'
# Such that ϵᵢϵ'ⱼ ≠ 0 for all i, j. A dual vector in these can be written as
#
# (a + Bϵ) + (C + Dϵ)ϵ'
#
# Where a is a value of reals, B is the matrix of coefficients of ϵ,
# C is the coefficients of ϵ' and D is the 3-tensor of coefficients of ϵϵ',
# (one dimension for each of a, ϵ and ϵ'. This is expanded by applying D to ϵ
# to first obtain a matrix via tensor contraction, and then applying this to ϵ')
#
# Analogously, if we fix B = I, C = I, D = 0 and apply a function f: ℝⁿ → ℝᵐ
# the coefficient of ϵϵ' will be an n x n x m tensor T such that T[:, :, i]
# is the Hessian of the i-th result of f. More simply, if we consider a 
# function f: ℝⁿ → ℝ, the coefficient of ϵϵ' will be the Hessian of f.
#
# In DualArrays.jl, this can be achieved by setting up a DualVector with a DualVector
# value and DualMatrix jacobian. Applying f: ℝⁿ → ℝ will give us a Dual with a Dual
# value and a DualVector vector partials. The jacobian of the DualVector partials
# is our Hessian.
#
# We now implement an example using f(x) = x[1] * x[2] below.

using DualArrays, SparseArrayKit
f(x) = x[1] * x[2]

# a + Bϵ
x = DualVector([2, 3], [1 0; 0 1])

# C + Dϵ
y = DualMatrix([1 0;0 1], SparseArray{Float64, 3}(undef, (2, 2, 2)))

# Setup the nested dual vector (a + Bϵ) + (C + Dϵ)ϵ'
z = DualVector(x, y)

# This will return a Dual number:
# a + b^Tϵ + (c + Dϵ)ϵ'
result = f(z)

# equivalent to a
value = result.value.value
# equivalent to b
gradient = result.value.partials
# equivalent to c
gradient2 = result.partials.value
# equivalent to d
_hessian = result.partials.jacobian.data

# We expect this to be 6
print("Value: ", value, "\n")
# We expect this to be [3, 2]
print("Gradient: ", gradient, "\n")
# We expect this to be the same
print("Gradient: ", gradient2, "\n")
# We expect this to be [0 1; 1 0]
print("Hessian: ", _hessian, "\n")

