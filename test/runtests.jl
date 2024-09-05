using VLBILikelihoods
using Test
using ChainRulesTestUtils
using ChainRulesCore
using Distributions
using StaticArrays
using Zygote
using LinearAlgebra
using FiniteDifferences
using StructArrays
using SparseArrays
using Serialization
using ComradeBase

function moment_test(d, nsamples=200_000, atol=5e-2)
    # c = cov(d)
    s = reduce(hcat, reshape.(rand(d, nsamples), :))
    cs = cov(s; dims=2)
    ms = reshape(mean(s; dims=2), size(d))
    # @test isapprox(c, cs; atol)
    @test isapprox(mean(d), ms; atol)

end

function lklhd_moment_test(d, nsamples=1_000_000, atol=5e-2)
    # c = cov(d)
    s = rand(d, nsamples)
    cs = vec(var(s; dims=2))
    ms = reshape(mean(s; dims=2), :)
    @test isapprox(var(d), cs; atol)
    @test isapprox(mean(d), ms; atol)

end


@testset "VLBILikelihoods.jl" begin

    @testset "utility test" begin

        @testset "diagonal normal stuff" begin
            μ = rand(50)
            Σ = rand(50)
            x = rand(50)
            test_rrule(VLBILikelihoods._unnormed_logpdf_μΣ, μ, Σ, x)
            @inferred Zygote.gradient(VLBILikelihoods._unnormed_logpdf_μΣ, μ, Σ, x)
        end

        @testset "dense normal stuff" begin
            μ = rand(50)
            m = rand(50,50)
            Σ = 0.5.*(m .+ m') .+ 5 .* Diagonal(ones(50))
            d = AmplitudeLikelihood(μ, Σ)
            @inferred VLBILikelihoods._gaussnorm(μ, d.Σ)
            @inferred VLBILikelihoods._chi2(μ, Σ)
        end

        @testset "diagonal complex normal stuff" begin
            # Now for complex numbers
            μc = rand(ComplexF64, 50)
            xc = rand(ComplexF64, 50)
            Σ = rand(50)
            test_rrule(VLBILikelihoods._unnormed_logpdf_μΣ, μc, Σ, xc)
            @inferred Zygote.gradient(VLBILikelihoods._unnormed_logpdf_μΣ, μc, Σ, xc)
        end



    end

    @testset "CholeskyFactor" begin
        m = sprand(50, 50, 0.01)
        Σ = 0.5.*(m .+ m') + 5 .* Diagonal(ones(50))
        c = VLBILikelihoods.CholeskyFactor(Σ)
        @test parent(c) === Σ
        @test size(Σ) == size(c)
        @test getindex(Σ, 1, 1) == getindex(c, 1, 1)
        @test cholesky(c) === c

        serialize("test.jls", c)
        c2 = deserialize("test.jls")
        rm("test.jls")
        x = rand(size(c2, 1))
        @test c2.cov == c.cov
        @test c2\x == c\x
    end

    include(joinpath(@__DIR__, "amplitude.jl"))
    include(joinpath(@__DIR__, "closure_phase.jl"))
    include(joinpath(@__DIR__, "complex_vis.jl"))
    include(joinpath(@__DIR__, "coherency.jl"))




end
