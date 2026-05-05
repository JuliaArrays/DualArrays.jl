using DualArrays, Test, LinearAlgebra, ForwardDiff, BandedMatrices
using DualArrays: ArrayOperator

@testset "DualArrays" begin
    
    @testset "Type Definition" begin
        @test_throws ArgumentError DualVector([1,2],I(3))
        @test Dual(1.0, [1, 2, 3]).partials == [1.0, 2.0, 3.0]
    end
    
    @testset "Indexing" begin
        v = DualVector([1., 2, 3], [1 2 3; 4 5 6;7 8 9])
        m = DualMatrix([1 2;3 4], zeros(2, 2, 2))

        @test size(v) == (3,)
        @test axes(v) == (Base.OneTo(3),)

        @test size(m) == (2, 2)
        @test axes(m) == (Base.OneTo(2), Base.OneTo(2))

        @test m[1, 1] == Dual(1, [0, 0])

        @test v[1] isa Dual
        @test v[1] == Dual(1,[1,2,3])
        @test v[2] == Dual(2,[4,5,6])
        @test v[3] == Dual(3,[7,8,9])
        @test_throws BoundsError v[4]
        @test v == DualVector([1,2, 3], [1 2 3; 4 5 6;7 8 9])

        x,y = v[1:2],v[2:3]
        @test x == DualVector([1,2],[1 2 3;4 5 6])
        @test y == DualVector([2,3],[4 5 6;7 8 9])

        n = 10
        v = DualVector(1:n, I(n))
        @test v[2:end].jacobian.data isa BandedMatrix

        @test sum(v[1:end-1] .* v[2:end]).partials == ForwardDiff.gradient(v -> sum(v[1:end-1] .* v[2:end]), 1:n)
    end

    @testset "Indexing (DualMatrix)" begin
        M = DualMatrix([1 2; 3 4], zeros(2, 2, 2, 2))
        @test M[1, 1] == Dual(1, [0, 0, 0, 0])
        @test M[1, :] == DualVector([1, 2], zeros(2, 4))
    end
    
    @testset "Arithmetic (DualVector)" begin
        v = DualVector([1, 2, 3], [1 2 3; 4 5 6;7 8 9])
        w = v + v
        @test w == DualVector([2,4,6],[2 4 6;8 10 12;14 16 18])
        @test w.jacobian == 2v.jacobian

        x = Dual(1, [1, 2, 3])
        y = DualVector([2, 3], [4 5 6;7 8 9])

        @test x .* y == DualVector([2,3],[6 9 12;10 14 18])
        
        @test sum(x .* y) isa Dual
        @test sum(x .* y) == Dual(5,[16,23,30])
    end

    @testset "Arithmetic (Dual)" begin
        a = Dual(2., [1, 2, 3.])
        b = Dual(3., [4, 5, 6.])

        @test b % 2 == Dual(1., [4, 5, 6.])
        @test 4 / a == Dual(2., [-1, -2, -3.])

        @test a + b == Dual(5, [5, 7, 9])
        @test a - b == Dual(-1, [-3, -3, -3])
        @test a * b == Dual(6, [11, 16, 21])
        @test isapprox(a / b, Dual(2/3, [(3*1 - 2*4)/9, (3*2 - 2*5)/9, (3*3 - 2*6)/9]))
        @test a ^ 3 == Dual(8, [12, 24, 36])
        @test b % a == Dual(1, [3, 3, 3])

        @test sin(a) == Dual(sin(2), cos(2) * [1, 2, 3])
        @test cos(b) == Dual(cos(3), -sin(3) * [4, 5, 6])
    end

    @testset "Dot product" begin
        v = DualVector([1, 2], [1 2; 3 4])
        w = DualVector([3, 4], [5 6; 7 8])
        @test dot(v, w) == Dual(11, [34, 44])
        @test dot(v, [0,1] ) == Dual(2, [3,4])
        @test dot([1,0], w) == Dual(3, [5,6])
    end

    @testset "Solve" begin
        A = [1 1; 1 -1]
        b = DualVector([2, 0], [3 4; 0 0])
        @test A \ b == DualVector([1, 1], [1.5 2; 1.5 2])
    end

    @testset "Matrix multiplication" begin
        M = [1 1; 1 1]
        d = DualVector([2, 3], [4 5; 6 7])
        @test M * d isa DualVector
        @test M * d == DualVector([5,5],[10 12;10 12])
    end
    @testset "vcat" begin
        x = Dual(1, [1, 2, 3])
        y = DualVector([2, 3], [4 5 6;7 8 9])
        @test vcat(x) == DualVector([1], [1 2 3])
        @test vcat(x, x) == DualVector([1, 1], [1 2 3;1 2 3])
        @test vcat(x, y) == DualVector([1, 2, 3], [1 2 3;4 5 6;7 8 9])

        z = DualVector([2, 3], [1 0; 0 1])
        @test vcat(1, z).jacobian.data == [0 0; 1 0; 0 1]
        @test vcat(z, 1).jacobian.data == [1 0; 0 1; 0 0]
    end

    @testset "show" begin
        d = DualVector([1.0, 2.0], [1 0; 0 1])
        s = repr(MIME"text/plain"(), d)
        @test occursin(" + ", s)
        @test endswith(s, "𝛜")
    end

    @testset "Nested Dual Numbers" begin
        # Example of finding Hessian of f(x) = x[1]*x[2]
        # Through nested dual numbers
        d = DualVector([1, 2], sparse(I, 2, 2))
        D = DualMatrix(sparse(I, 2, 2), zeros(2, 2, 2, 1))
        d2 = DualVector(d, D)
        val = d2[1] * d2[2]
        @test val isa Dual
        @test val.value isa Dual
        @test val.partials isa DualVector

        @test val.value.value == 2 # f(x)
        @test val.value.partials == [2, 1] # ∇f(x)
        @test val.partials.value == [2, 1] # ∇f(x)
        @test val.partials.jacobian == [0 1; 1 0] # Hessian of f(x)
    end
    
    include("broadcast_test.jl")
    include("array_operator_test.jl")
end