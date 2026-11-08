@testset "Amplitude Likelihood" begin
    rng = Random.Xoshiro(3)
    n = 50

    @testset "weights" begin
        μ = rand(rng, n)
        w = inv.(rand(rng, n) .+ 1.0)
        d = AmplitudeLikelihood(μ, w)
        x = rand(rng, d)
        @test logdensityof(d, x) ≈ logpdf(MvNormal(μ, Diagonal(inv.(w))), x)
        @inferred logdensityof(d, x)
        s = stack(rand(rng, d) for _ in 1:50_000)
        @test vec(std(s; dims = 2)) ≈ inv.(sqrt.(w)) rtol = 3.0e-2

        w0 = copy(w)
        w0[1:5] .= 0
        x0 = copy(x)
        x0[1:5] .= 0
        @test logdensityof(AmplitudeLikelihood(μ, w0), x0) ≈ logdensityof(AmplitudeLikelihood(μ[6:end], w[6:end]), x[6:end])
        check_gradient(m -> logdensityof(AmplitudeLikelihood(m, w, d.lognorm), x), μ)
    end

    @testset "covariance" for Σ in (randpd(n), Matrix(randpd(n)))
        μ = rand(rng, n)
        d = AmplitudeLikelihood(μ, CholeskyFactor(Σ))
        x = rand(rng, d)
        @test logdensityof(d, x) ≈ logpdf(MvNormal(μ, Matrix(Σ)), x)
        s = stack(rand(rng, d) for _ in 1:20_000)
        @test within(vec(mean(s; dims = 2)), μ, sqrt.(diag(Matrix(Σ))), size(s, 2))
        @test_throws "CholeskyFactor" AmplitudeLikelihood(μ, Σ)
        check_gradient(m -> logdensityof(AmplitudeLikelihood(m, d.w, d.lognorm), x), μ)
    end
end

@testset "Rice Amplitude Likelihood" begin
    rng = Random.Xoshiro(4)
    n = 50
    μ = rand(rng, n)
    w = fill(inv(5.0), n)
    d = RiceAmplitudeLikelihood(μ, w)
    x = rand(rng, d)

    s = stack(rand(rng, d) for _ in 1:20_000)
    @test vec(mean(s; dims = 2)) ≈ mean(d) rtol = 5.0e-2
    @test vec(var(s; dims = 2)) ≈ var(d) rtol = 1.0e-1
    @test logdensityof(d, -x) == -Inf

    @testset "unmeasured values" begin
        w0 = copy(w)
        w0[1:5] .= 0
        x0 = copy(x)
        x0[1:5] .= 0
        @test logdensityof(RiceAmplitudeLikelihood(μ, w0), x0) ≈ logdensityof(RiceAmplitudeLikelihood(μ[6:end], w[6:end]), x[6:end])
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(m -> logdensityof(RiceAmplitudeLikelihood(m, w0, 0.0), x0)), μ)[1]
        @test all(isfinite, g)
        @test all(iszero, g[1:5])
    end

    @testset "high SNR" begin
        μh = 100 * rand(rng, n) .+ 2.0
        wh = inv.(1.0e-6 * rand(rng, n))
        dh = RiceAmplitudeLikelihood(μh, wh)
        xh = rand(rng, dh)
        @test logdensityof(dh, xh) ≈ logpdf(MvNormal(μh, Diagonal(inv.(wh))), xh) atol = 1.0e-3
    end

    check_gradient(m -> logdensityof(RiceAmplitudeLikelihood(m, w, d.lognorm), x), μ)
end
