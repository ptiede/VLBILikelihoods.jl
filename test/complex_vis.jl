@testset "Complex Vis" begin

    μR = rand(Float32, 100)
    μI = rand(Float32, 100)
    μ = μR .+ 1im.*μI
    xR = rand(Float32, 100)
    xI = rand(Float32, 100)

    Σ = 2*rand(Float32, 100) .+ 1f0
    d32 = ComplexVisLikelihood(μ, Σ)

    @inferred Float32 logdensityof(d32, μ)


    μR = rand(100)
    μI = rand(100)
    μ = μR .+ 1im.*μI
    xR = rand(100)
    xI = rand(100)

    Σ = 0.5.*rand(100) .+ 1.0

    d = ComplexVisLikelihood(μ, Σ)
    x = rand(d)
    xR = real.(x)
    xI = imag.(x)
    dR = MvNormal(μR, Diagonal(Σ))
    dI = MvNormal(μI, Diagonal(Σ))

    @inferred Float64 logdensityof(d, x)
    @test logdensityof(d, x) ≈ logpdf(dR, xR) + logpdf(dI, xI)


    lklhd_moment_test(d, 1_000_000, 5e-2)

    f(x, μ, Σ) = logdensityof(ComplexVisLikelihood(μ, Σ), x)
    # @inferred Zygote.gradient(f, x, μ, Σ)
    test_rrule(ComplexVisLikelihood, μ, Σ)
end
