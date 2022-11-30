using VLBILikelihoods
using Test
using ChainRulesTestUtils
using ChainRulesCore
using Distributions
using StaticArrays
using Zygote
using LinearAlgebra

@testset "VLBILikelihoods.jl" begin

    include(joinpath(@__DIR__, "amplitude.jl"))
    include(joinpath(@__DIR__, "coherency.jl"))




end
