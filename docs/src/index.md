# DualArrays.jl

Documentation for `DualArrays.jl`.

## Dual

```@docs
DualArrays.Dual
```

```@example
using DualArrays

a = Dual(2.0, [1.0, 2.0, 3.0])
b = Dual(3.0, [4.0, 5.0, 6.0])

a * b
```

## DualArray

```@docs
DualArrays.DualArray
DualArrays.DualVector
DualArrays.DualMatrix
```

```@example
using DualArrays

v = DualVector([1.0, 2.0, 3.0], [1 2 3; 4 5 6; 7 8 9])
v[2]
```

## ArrayOperator

```@docs
DualArrays.ArrayOperator
```

### Broadcasting with ArrayOperators

```@eval
using DualArrays
import Base.Docs
import Markdown

Markdown.parse(sprint(show, Docs.doc(DualArrays.ArrayOperatorBroadcastStyle)))
```

```@example
using DualArrays

t = ArrayOperator{1}([1 2 3; 4 5 6; 7 8 9])

t .+ 1
```

### Multiplication with ArrayOperators

```@eval
using DualArrays
import Base.Docs
import Markdown

Markdown.parse(sprint(show, Docs.doc(DualArrays._contract)))
```

```@example
using DualArrays

t = ArrayOperator{1}([1 2 3; 4 5 6; 7 8 9])

t * t
```

## Utilities

```@docs
DualArrays.jacobian
```

```@example
using DualArrays

f(x) = sin.(x) .+ x .* x

J = jacobian(f, [1.0, 2.0, 3.0])
J.data
```
