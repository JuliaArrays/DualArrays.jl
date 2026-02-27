using DualArrays, Test, SparseArrays, LinearAlgebra

@testset "Broadcasting" begin
    d = DualVector([1.0, 2.0, 3.0], [1 0 0; 0 1 0; 0 0 1])

    # Test broadcasting with single argument functions
    s = sin.(d)
    c = cos.(d)
    e = exp.(d)
    l = log.(d)
    a = tanh.(d)

    @test s isa DualVector
    @test c isa DualVector
    @test e isa DualVector
    @test l isa DualVector
    @test a isa DualVector

    @test s.value ≈ sin.([1.0, 2.0, 3.0])
    @test c.value ≈ cos.([1.0, 2.0, 3.0])
    @test e.value ≈ exp.([1.0, 2.0, 3.0])
    @test l.value ≈ log.([1.0, 2.0, 3.0])
    @test a.value ≈ tanh.([1.0, 2.0, 3.0])

    @test s.jacobian ≈ Diagonal(cos.([1.0, 2.0, 3.0]))
    @test c.jacobian ≈ Diagonal(-sin.([1.0, 2.0, 3.0]))
    @test e.jacobian ≈ Diagonal(exp.([1.0, 2.0, 3.0]))
    @test l.jacobian ≈ Diagonal(1.0 ./ [1.0, 2.0, 3.0])
    @test a.jacobian ≈ Diagonal(1.0 .- tanh.([1.0, 2.0, 3.0]).^2)

    # Test broadcasting with binary operations
    a = d .+ 2.0
    s = 3.0 .- d
    m = d .* 4.0
    div = 8.0 ./ d

    @test a isa DualVector
    @test s isa DualVector 
    @test m isa DualVector
    @test div isa DualVector

    @test a.value ≈ [3.0, 4.0, 5.0]
    @test s.value ≈ [2.0, 1.0, 0.0]
    @test m.value ≈ [4.0, 8.0, 12.0]
    @test div.value ≈ [8.0, 4.0, 8/3.0]

    @test a.jacobian ≈ d.jacobian
    @test s.jacobian ≈ -d.jacobian
    @test m.jacobian ≈ 4.0 * d.jacobian
    @test div.jacobian ≈ (-8.0 ./ (d.value .^ 2)) .* d.jacobian

    # Test broadcasting with binary operations on a Dual and DualVector
    x = Dual(2.0, [1.0, 2.0, 3.0])

    a = x .+ d
    s = d .- x
    m = x .* d
    div = d ./ x

    @test a isa DualVector
    @test s isa DualVector
    @test m isa DualVector
    @test div isa DualVector

    @test a.value ≈ [3.0, 4.0, 5.0]
    @test s.value ≈ [-1.0, 0.0, 1.0]
    @test m.value ≈ [2.0, 4.0, 6.0]
    @test div.value ≈ [0.5, 1.0, 1.5]

    @test a.jacobian ≈ [2.0 2.0 3.0; 1.0 3.0 3.0; 1.0 2.0 4.0]
    @test s.jacobian ≈ [0.0 -2.0 -3.0; -1.0 -1.0 -3.0; -1.0 -2.0 -2.0]
    @test m.jacobian ≈ [3.0 2.0 3.0; 2.0 6.0 6.0; 3.0 6.0 11.0]
    @test div.jacobian ≈ [0.25 -0.5 -0.75; -0.5 -0.5 -1.5; -0.75 -1.5 -1.75]

    # Broadcasting between Dual and AbstractVector
    a = x .+ [1.0, 2.0, 3.0]
    s = [1.0, 2.0, 3.0] .- x
    m = x .* [1.0, 2.0, 3.0]
    div = [1.0, 2.0, 3.0] ./ x

    @test a isa DualVector
    @test s isa DualVector
    @test m isa DualVector
    @test div isa DualVector

    @test a.value ≈ [3.0, 4.0, 5.0]
    @test s.value ≈ [-1.0, 0.0, 1.0]
    @test m.value ≈ [2.0, 4.0, 6.0]
    @test div.value ≈ [0.5, 1.0, 1.5]

    @test a.jacobian ≈ [1.0 2.0 3.0; 1.0 2.0 3.0; 1.0 2.0 3.0]
    @test s.jacobian ≈ [-1.0 -2.0 -3.0; -1.0 -2.0 -3.0; -1.0 -2.0 -3.0]
    @test m.jacobian ≈ [1.0 2.0 3.0; 2.0 4.0 6.0; 3.0 6.0 9.0]
    @test div.jacobian ≈ [-0.25 -0.5 -0.75; -0.5 -1.0 -1.5; -0.75 -1.5 -2.25]
end