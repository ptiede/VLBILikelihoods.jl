@testset "Closure Phase" begin
    rng = Random.Xoshiro(5)
    n = 50

    @testset "weights" begin
        μ = rand(rng, n)
        w = inv.(rand(rng, n) .+ 0.1)
        d = ClosurePhaseLikelihood(μ, w)
        @test VLBILikelihoods.unnormed_logpdf(d, μ) ≈ 0.0
        # independent von Mises distributions with concentration w
        x = rand(rng, d)
        @test logdensityof(d, x) ≈ sum(logpdf.(VonMises.(μ, w), x))
        @inferred logdensityof(d, x)

        w0 = copy(w)
        w0[1:5] .= 0
        @test logdensityof(ClosurePhaseLikelihood(μ, w0), x) ≈ logdensityof(ClosurePhaseLikelihood(μ[6:end], w[6:end]), x[6:end])
        check_gradient(m -> logdensityof(ClosurePhaseLikelihood(m, w, d.lognorm), x), μ)
    end

    @testset "covariance" for Σ in (randpd(n), Matrix(randpd(n)))
        μ = rand(rng, n)
        d = ClosurePhaseLikelihood(μ, CholeskyFactor(Σ))
        @test VLBILikelihoods.unnormed_logpdf(d, μ) ≈ 0.0
        s = stack(rand(rng, d) for _ in 1:20_000)
        @test within(vec(mean(s; dims = 2)), μ, sqrt.(diag(Matrix(Σ))), size(s, 2))
        @test_throws "CholeskyFactor" ClosurePhaseLikelihood(μ, Σ)
        x = rand(rng, d)
        check_gradient(m -> logdensityof(ClosurePhaseLikelihood(m, d.w, d.lognorm), x), μ)
    end
end
