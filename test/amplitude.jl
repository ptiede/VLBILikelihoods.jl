@testset "Amplitude Likelihood" begin

    m = central_fdm(5,1)

    μ = rand(50)
    σ = rand(50,50)
    Σ = 0.5.*(σ .+ σ') .+ 5 .* Diagonal(ones(50))
    Σd = diag(Σ)

    dv = AmplitudeLikelihood(μ, Σ)
    dv2 = AmplitudeLikelihood(μ, Σd)
    dd = MvNormal(μ, Σ)
    dd2 = MvNormal(μ, Diagonal(Σd))

    x = rand(dv)
    @test all(isapprox.(mean(rand(dv, 10000),dims=2), μ; atol=5*sqrt(maximum(Σ))/(sqrt(10_000))))

    @test logpdf(dv, x) ≈ logpdf(dd, x)
    @test logpdf(dv2, x) ≈ logpdf(dd2, x)

    test_rrule(AmplitudeLikelihood, μ, Σd)

    @inferred Zygote.gradient(logdensityof(dv), x)
    @inferred Zygote.gradient(logdensityof(dv2), x)

    f(x, μ, Σ) = logdensityof(AmplitudeLikelihood(μ, Σ), x)


    gvz  = Zygote.gradient(f, x, μ, Σd)
    gvz2 = Zygote.gradient(f, x, μ, Diagonal(Σ))
    @inferred Zygote.gradient(f, x, μ, Σ)

    gfdz  = grad(m, f, x, μ, Σd)
    @test all(isapprox.(gvz, gfdz))

end
