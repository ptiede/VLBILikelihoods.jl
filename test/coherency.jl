@testset "Coherency Likelihood" begin
    m = central_fdm(5,1)


    μRR = rand(ComplexF64, 10)
    μRL = rand(ComplexF64, 10)
    μLR = rand(ComplexF64, 10)
    μLL = rand(ComplexF64, 10)

    xRR = rand(ComplexF64, 10)
    xRL = rand(ComplexF64, 10)
    xLR = rand(ComplexF64, 10)
    xLL = rand(ComplexF64, 10)

    ΣRR = rand(Float64, 10) .+ 3.0
    ΣRL = rand(Float64, 10) .+ 3.0
    ΣLR = rand(Float64, 10) .+ 3.0
    ΣLL = rand(Float64, 10) .+ 3.0

    μ = StructVector(SMatrix{2,2}.(μRR, μLR, μRL, μLL))
    x = StructVector(SMatrix{2,2}.(xRR, xLR, xRL, xLL))
    Σ = StructVector(SMatrix{2,2}.(ΣRR, ΣLR, ΣRL, ΣLL))

    d = CoherencyLikelihood(μ, Σ)
    darr = CoherencyLikelihood(Array(μ), Array(Σ))

    @test d.lognorm ≈ darr.lognorm
    @test logdensityof(d, x) ≈ logdensityof(darr, Array(x))

    dRR_r = MvNormal(real.(μRR), Diagonal(ΣRR))
    dLL_r = MvNormal(real.(μLL), Diagonal(ΣLL))
    dRL_r = MvNormal(real.(μRL), Diagonal(ΣRL))
    dLR_r = MvNormal(real.(μLR), Diagonal(ΣLR))

    dRR_i = MvNormal(imag.(μRR), Diagonal(ΣRR))
    dLL_i = MvNormal(imag.(μLL), Diagonal(ΣLL))
    dRL_i = MvNormal(imag.(μRL), Diagonal(ΣRL))
    dLR_i = MvNormal(imag.(μLR), Diagonal(ΣLR))

    @test logdensityof(d, x) ≈ logpdf(dRR_r, real.(xRR)) + logpdf(dRR_i, imag.(xRR)) +
                               logpdf(dLL_r, real.(xLL)) + logpdf(dLL_i, imag.(xLL)) +
                               logpdf(dRL_r, real.(xRL)) + logpdf(dRL_i, imag.(xRL)) +
                               logpdf(dLR_r, real.(xLR)) + logpdf(dLR_i, imag.(xLR))


    test_rrule(Zygote.ZygoteRuleConfig(), unnormed_logpdf, d, x)
    test_rrule(VLBILikelihoods._coherencynorm, μ, Σ)
    ll(d, x) = unnormed_logpdf(d, x)
    ll(μ, Σ, x) = unnormed_logpdf(CoherencyLikelihood(μ, Σ), x)

    @inferred ll(d, x)
end
