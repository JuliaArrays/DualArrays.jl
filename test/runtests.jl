using DualArrays, Test, LinearAlgebra, ForwardDiff, BandedMatrices
using DualArrays: Dual

@testset "DualArrays" begin
    
    @testset "Type Definition" begin
        @test_throws ArgumentError DualVector([1,2],I(3))
    end
    
    @testset "Indexing" begin
        v = DualVector([1, 2, 3], [1 2 3; 4 5 6;7 8 9])
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
        @test v[2:end].jacobian isa BandedMatrix

        @test sum(v[1:end-1] .* v[2:end]).partials == ForwardDiff.gradient(v -> sum(v[1:end-1] .* v[2:end]), 1:n)
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
    end

    @testset "Hessian" begin
        d = DualVector(
            DualVector([1, 2], [1 0;0 1]),
            [1 0;0 1]
        )
        f(x) = x[1] * x[2]
        @test f(d) isa Dual
        @test f(d).partials isa DualVector
        @test f(d).partials.jacobian == [0 1; 1 0]
    end
    
    include("broadcast_test.jl")
end