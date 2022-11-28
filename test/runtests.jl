using VLBILikelihoods
using Test

#@testset "VLBILikelihoods.jl" begin


    μRR = rand(ComplexF64, 1000)
    μRL = rand(ComplexF64, 1000)
    μLR = rand(ComplexF64, 1000)
    μLL = rand(ComplexF64, 1000)

    xRR = rand(ComplexF64, 1000)
    xRL = rand(ComplexF64, 1000)
    xLR = rand(ComplexF64, 1000)
    xLL = rand(ComplexF64, 1000)

    ΣRR = rand(Float64, 1000)
    ΣRL = rand(Float64, 1000)
    ΣLR = rand(Float64, 1000)
    ΣLL = rand(Float64, 1000)

    μ = SMatrix{2,2}.(μRR, μLR, μRL, μLL)
    x = SMatrix{2,2}.(xRR, xLR, xRL, xLL)
    Σ = SMatrix{2,2}.(ΣRR, ΣLR, ΣRL, ΣLL)

    d = CoherencyLikelihood(μ, Σ)

    ll(μ, Σ, x) = VLBILikelihoods.unnormed_logpdf(CoherencyLikelihood(μ, Σ), x)


#end
