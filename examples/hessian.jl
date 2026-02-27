using DualArrays, StaticArrays
using DualArrays: Dual

f = x -> exp(cos(x))
fp = x -> f(Dual(x,[1])).partials

fp(Dual(0.1, [1]))

f = x -> cos.(x)[1] * exp.(x)[2]
leftone(x::Vector{T}) where {T} = convert(Matrix{T}, I(length(x)))
# leftone(x::DualVector{T}) where {T} = DualMatrix(convert(Matrix{T}, I(length(x))),

∇f = x -> f(DualVector(x, leftone(x))).partials

let (x,y) = (0.1,0.2)
    @test ∇f([x,y]) ≈ [-sin(x)exp(y), cos(x)*exp(y)]
    @test vcat(transpose.(getfield.(∇f(DualVector([x,y], [1 0; 0 1])), :partials))...) ≈ [-cos(x)exp(y) -sin(x)exp(y); -sin(x)exp(y) cos(x)exp(y)]
end