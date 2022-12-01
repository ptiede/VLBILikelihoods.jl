@testset "Amplitude Likelihood" begin
    μ = rand(50)
    Σ = rand(50)

    dv = AmplitudeLikelihood(μ, Σ)
    dv2 = AmplitudeLikelihood(μ, Diagonal(Σ))
    dd = MvNormal(μ, Diagonal(Σ))

    x = rand(dv)
    @test all(isapprox.(mean(rand(dv, 10000),dims=2), μ; atol=5*sqrt(maximum(Σ))/(sqrt(10_000))))

    @test logpdf(dv, x) ≈ logpdf(dd, x)
    @test logpdf(dv2, x) ≈ logpdf(dd, x)

    test_rrule(AmplitudeLikelihood, μ, Σ)

    @inferred
end
