@testset "Complex Vis" begin
    μR = rand(100)
    μI = rand(100)
    μ = μR .+ 1im.*μI
    xR = rand(100)
    xI = rand(100)

    Σ = 2*rand(100) .+ 1.0

    d = ComplexVisLikelihood(μ, Σ)
    x = rand(d)
    xR = real.(x)
    xI = imag.(x)
    dR = MvNormal(μR, Diagonal(Σ))
    dI = MvNormal(μI, Diagonal(Σ))


    @test logdensityof(d, x) ≈ logpdf(dR, xR) + logpdf(dI, xI)

    f(x, μ, Σ) = logdensityof(ComplexVisLikelihood(μ, Σ), x)
    @inferred Zygote.gradient(f, x, μ, Σ)
    test_rrule(ComplexVisLikelihood, μ, Σ)
end
