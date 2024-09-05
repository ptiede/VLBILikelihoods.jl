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
    g = UnstructuredDomain((U=randn(10), V=randn(10)))
    μs = UnstructuredMap(μ, g)
    x = StructVector(SMatrix{2,2}.(xRR, xLR, xRL, xLL))
    Σ = StructVector(SMatrix{2,2}.(ΣRR, ΣLR, ΣRL, ΣLL))
    Σs= UnstructuredMap(Σ, g)
    d = CoherencyLikelihood(μ, Σ)
    ds1 = CoherencyLikelihood(μs, Σ)
    d1s = CoherencyLikelihood(μ, Σs)
    dss = CoherencyLikelihood(μs, Σs)
    darr = CoherencyLikelihood(Array(μ), Array(Σ))

    @test typeof(ds1.μ) == typeof(d.μ)
    @test typeof(d1s.μ) == typeof(d.μ)
    @test typeof(dss.μ) == typeof(d.μ)
    @test typeof(ds1.Σ) == typeof(d.Σ)
    @test typeof(d1s.Σ) == typeof(d.Σ)
    @test typeof(dss.Σ) == typeof(d.Σ)

    @test eltype(d) == eltype(μ)
    @test length(d) == length(μ)

    @test d.lognorm ≈ darr.lognorm
    @test logdensityof(d, x) ≈ logdensityof(darr, Array(x))
    @inferred Float64 logdensityof(d, x)

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

    rand(d)
    rand(d, 1)
    @test size(rand(d, 2,3)) == (length(d.μ), 2,3)

    s = rand(d, 1_000_000)
    m = mean(s, dims=2)
    v1 = var(s.:1, dims=2)
    v2 = var(s.:2, dims=2)
    v3 = var(s.:3, dims=2)
    v4 = var(s.:4, dims=2)
    Σmax = mapreduce(maximum, max, d.Σ)
    @test isapprox(m, d.μ, atol = 5e-2)
    @test isapprox(v1, d.Σ.:1, atol = 5e-2)
    @test isapprox(v2, d.Σ.:2, atol = 5e-2)
    @test isapprox(v3, d.Σ.:3, atol = 5e-2)
    @test isapprox(v4, d.Σ.:4, atol = 5e-2)

    @inferred ll(d, x)
end
