using Test, LinearAlgebra
using DualArrays: ArrayOperator
using FillArrays

@testset "ArrayOperator" begin
    t = ArrayOperator{1}([1 2 3;4 5 6;7 8 9])

    @testset "eltype" begin
        @test eltype(t) == eltype(typeof(t)) == Int
    end

    @testset "Equality" begin
        @test t == ArrayOperator{1}([1 2 3;4 5 6;7 8 9])
        @test isapprox(t, ArrayOperator{1}([1 2 3;4 5 6;7 8 9]))
        @test t != [1 2 3;4 5 6;7 8 9]
        @test t != ArrayOperator{0}([1 2 3;4 5 6;7 8 9])
    end
    @testset "Indexing" begin
        @test t[1,1] == 1
        @test t[(1,), (:,)] == ArrayOperator{0}([1, 2, 3])
        @test t[1:2, 1:2] == ArrayOperator{1}([1 2;4 5])

        s = ArrayOperator{2}(ones(2,2,2))
        @test s[(1,1), (:,)] == ArrayOperator{0}([1, 1])
        @test s[(1,:), (:,)] == ArrayOperator{1}([1 1;1 1])
        @test s[(1,2), :] == ArrayOperator{0}([1, 1])
    end

    @testset "Arithmetic" begin
        @test t + t isa ArrayOperator
        @test t + t == ArrayOperator{1}([2 4 6;8 10 12;14 16 18])

        @test t .* 2 isa ArrayOperator
        @test t .* 2 == ArrayOperator{1}([2 4 6;8 10 12;14 16 18])
        @test t .* [1,2,3] == ArrayOperator{1}([1 2 3;8 10 12;21 24 27])

        t1 = ArrayOperator{0}([1, 2, 3])
        t2 = ArrayOperator{1}([1 2 3])
        t3 = ArrayOperator{0}([2])

        @test_throws ArgumentError t1 .+ ones(3, 3)
        @test t1 .* t2 == ArrayOperator{1}([1 2 3;2 4 6;3 6 9])
        @test t1 .* t2 .+ t3 == ArrayOperator{1}([3 4 5;4 6 8;5 8 11])

        @test sin.(t1) == ArrayOperator{0}(sin.([1, 2, 3]))

        s = ArrayOperator{1}(zeros(3, 3))

        result = (s .= t .+ 1)

        @test result isa ArrayOperator
        @test result.data == s.data
        @test s == ArrayOperator{1}([2 3 4; 5 6 7; 8 9 10])
        @test t * t == ArrayOperator{1}([30 36 42;66 81 96; 102 126 150])
        @test t * [1, 0, 0] == ArrayOperator{1}([1,4,7])
        @test [1 0 0] * t == ArrayOperator{1}([1 2 3])

        left = ArrayOperator{1}([1 2; 3 4])
        right = ArrayOperator{2}(ones(2,2,2))
        expected = cat((left.data * right.data[:, :, k] for k in axes(right.data, 3))..., dims=3)
        @test left * right == ArrayOperator{2}(expected)

        @test t * 2 == ArrayOperator{1}([2 4 6;8 10 12;14 16 18])
        @test 2 * t == ArrayOperator{1}([2 4 6;8 10 12;14 16 18])

        @test s - t == ArrayOperator{1}(ones(3, 3))

        @test_throws ArgumentError ArrayOperator{1}(ones(3,3)) .* ArrayOperator{0}(ones(3,3))
    end

    @testset "Transpose" begin
        @test transpose(t) == ArrayOperator{1}([1 4 7;2 5 8;3 6 9])
        @test transpose(ArrayOperator{0}([1, 2, 3])) == ArrayOperator{1}([1 2 3])
    end

    @testset "Arithmetic with nonstandard array" begin
        o = OneElement(1, 1, 3)
        a = ArrayOperator{1}(o)
        b = ArrayOperator{1}(zeros(3))
        b .= a .* 3
        @test b == ArrayOperator{1}(OneElement(3, 1, 3))
    end

    @testset "Arithmetic with BandedMatrix" begin
        bm1 = BandedMatrix(0 => [1, 2, 3], 1 => [1, 1], -1 => [1, 1])
        bm2 = BandedMatrix(0 => [4, 5, 6], 1 => [1, 1], -1 => [1, 1])
        a = ArrayOperator{1}(bm1)
        b = ArrayOperator{1}(bm2)
        @test (a .+ b).data == bm1 .+ bm2
        @test (a .+ b).data isa BandedMatrix
    end
end