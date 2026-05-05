using Test, LinearAlgebra
using DualArrays: ArrayOperator

@testset "ArrayOperator" begin
    t = ArrayOperator{1}([1 2 3;4 5 6;7 8 9])

    @testset "Equality" begin
        @test t == [1 2 3;4 5 6;7 8 9]
        @test [1 2 3;4 5 6;7 8 9] == t
        @test isapprox(t, [1 2 3;4 5 6;7 8 9])
        @test isapprox([1 2 3;4 5 6;7 8 9], t)
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

        @test t1 .+ ones(3, 3) == [2 2 2;3 3 3; 4 4 4]
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

        @test t * 2 == [2 4 6;8 10 12;14 16 18]
        @test 2 * t == [2 4 6;8 10 12;14 16 18]

        @test s - t == ones(3, 3)
    end
end