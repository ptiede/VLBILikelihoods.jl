using VLBILikelihoods
using Test
using ChainRulesTestUtils
using ChainRulesCore
using Distributions
using StaticArraysCore
using Zygote
using LinearAlgebra
using FiniteDifferences

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

        # Now for complex numbers
        μc = rand(ComplexF64, 50)
        xc = rand(ComplexF64, 50)
        Σ = rand(50)
        test_rrule(VLBILikelihoods._unnormed_logpdf_μΣ, μc, Σ, xc)
        @inferred Zygote.gradient(VLBILikelihoods._unnormed_logpdf_μΣ, μc, Σ, xc)



    end

    include(joinpath(@__DIR__, "amplitude.jl"))
    include(joinpath(@__DIR__, "coherency.jl"))




end
