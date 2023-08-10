function amplitude_test(μ, Σ)
    m = central_fdm(5,1)

    Σd = Array(diag(Σ))

    dv = AmplitudeLikelihood(μ, Σ)
    dv2 = AmplitudeLikelihood(μ, Σd)
    @inferred logdensityof(dv, rand(dv))
    @inferred logdensityof(dv2, rand(dv))

    dd = MvNormal(μ, Array(Σ))
    dd2 = MvNormal(μ, Diagonal(Σd))

    x = rand(dv)
    λmax = maximum(eigvals(Matrix(Σ)))
    @test all(isapprox.(mean(rand(dv, 10_000),dims=2), μ; atol=10*sqrt(λmax)/(sqrt(10_000))))
    @test all(isapprox.(std(rand(dv, 500_000),dims=2), sqrt.(Σd); atol=5e-1))
    @test all(isapprox.(mean(rand(dv2, 10_000),dims=2), μ; atol=5*sqrt(maximum(Σd))/(sqrt(10_000))))
    @test all(isapprox.(std(rand(dv2, 500_000),dims=2), sqrt.(Σd); atol=5e-2))

    @test logpdf(dv, x) ≈ logpdf(dd, x)
    @test logpdf(dv2, x) ≈ logpdf(dd2, x)

    test_rrule(AmplitudeLikelihood, μ, Σd)

    # @inferred Zygote.gradient(logdensityof(dv), x)
    # @inferred Zygote.gradient(logdensityof(dv2), x)

    f(x, μ, Σ) = logdensityof(AmplitudeLikelihood(μ, Σ), x)


    gvz  = Zygote.gradient(f, x, μ, Σd)
    gvz2 = Zygote.gradient(f, x, μ, Diagonal(Σ))
    # @inferred Zygote.gradient(f, x, μ, Σ)

    gfdz  = grad(m, f, x, μ, Σd)
    @test all(isapprox.(gvz, gfdz))
end

@testset "Amplitude Likelihood" begin
    @testset "Sparse" begin
        μ = rand(50)
        σ = sprand(50,50, 0.05)
        Σ = 0.5.*(σ + σ') + 5 .* Diagonal(ones(50))
        amplitude_test(μ, Σ)
    end

    @testset "Dense" begin
        μ = rand(50)
        σ = rand(50,50)
        Σ = 0.5.*(σ + σ') + 5 .* Diagonal(ones(50))
        amplitude_test(μ, Σ)
    end
end

@testset "Rice Amplitude Likelihood" begin


    μ = rand(50)
    Σ = 5 .* Diagonal(ones(50))
    Σd = diag(Σ)

    dv = RiceAmplitudeLikelihood(μ, Σ)
    dv2 = RiceAmplitudeLikelihood(μ, Σd)

    x = rand(dv)
    @test all(isapprox.(mean(rand(dv, 10_000),dims=2), mean(dv); atol=5*sqrt(maximum(Σ))/(sqrt(10_000))))

    @test logpdf(dv, x) ≈ logpdf(dv, x)
    @test logpdf(dv2, x) ≈ logpdf(dv2, x)

    test_rrule(VLBILikelihoods.unnormed_logpdf, dv, x)
    test_rrule(RiceAmplitudeLikelihood, μ, Σd)

    @testset "High SNR" begin
        μ = 100*rand(50) .+ 2.0
        Σ = 1e-6*rand(50)
        d = RiceAmplitudeLikelihood(μ, Σ)
        dN = MvNormal(μ, Diagonal(Σ))
        x = rand(d)
        @test isapprox(logpdf(d, x), logpdf(dN, x), atol=1e-3)
    end
end
