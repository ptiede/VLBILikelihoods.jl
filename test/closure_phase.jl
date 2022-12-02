@testset "Closure Phase" begin
    μ = rand(50)
    σ = rand(50,50)
    Σ = 0.5.*(σ .+ σ') .+ 5 .* Diagonal(ones(50))
    Σd = diag(Σ)

    dv = ClosurePhaseLikelihood(μ, Σ)
    dv2 = ClosurePhaseLikelihood(μ, Σd)
    @test VLBILikelihoods.unnormed_logpdf(dv, μ) ≈ 0.0
    @test VLBILikelihoods.unnormed_logpdf(dv2, μ) ≈ 0.0

    x = rand(dv)
    @test all(isapprox.(mean(rand(dv, 10000),dims=2), μ; atol=5*sqrt(maximum(Σ))/(sqrt(10_000))))
    @test all(isapprox.(mean(rand(dv2, 10000),dims=2), μ; atol=5*sqrt(maximum(Σ))/(sqrt(10_000))))


    @test logdensityof(dv2, μ) ≈ VLBILikelihoods._closurephasenorm(μ, dv2.Σ)

    f(x, μ, Σ) = logdensityof(ClosurePhaseLikelihood(μ, Σ), x)


    gvz  = Zygote.gradient(f, x, μ, Σd)
    gvz2 = Zygote.gradient(f, x, μ, Diagonal(Σ))
    @inferred Zygote.gradient(f, x, μ, Σ)

    gfdz  = grad(m, f, x, μ, Σd)
    @test all(isapprox.(gvz, gfdz))


    test_rrule(VLBILikelihoods._cp_logpdf, μ, Σd, x)
    test_rrule(VLBILikelihoods._closurephasenorm, μ, Σd)

    @inferred Zygote.gradient(logdensityof(dv), x)
end
