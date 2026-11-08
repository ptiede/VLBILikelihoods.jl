@testset "Complex Vis" begin
    rng = Random.Xoshiro(1)
    n = 100
    μ = complex.(rand(rng, n), rand(rng, n))
    w = inv.(0.5 .* rand(rng, n) .+ 1.0)
    d = ComplexVisLikelihood(μ, w)
    x = rand(rng, d)

    @testset "density" begin
        dR = MvNormal(real.(μ), Diagonal(inv.(w)))
        dI = MvNormal(imag.(μ), Diagonal(inv.(w)))
        @test logdensityof(d, x) ≈ logpdf(dR, real.(x)) + logpdf(dI, imag.(x))
        @inferred Float64 logdensityof(d, x)
        d32 = ComplexVisLikelihood(ComplexF32.(μ), Float32.(w))
        @inferred Float32 logdensityof(d32, ComplexF32.(x))
    end

    @testset "unmeasured values" begin
        w0 = copy(w)
        w0[1:10] .= 0
        d0 = ComplexVisLikelihood(μ, w0)
        dsub = ComplexVisLikelihood(μ[11:end], w[11:end])
        x0 = copy(x)
        x0[1:10] .= 0
        @test logdensityof(d0, x0) ≈ logdensityof(dsub, x[11:end])
        @test all(==(μ[1]), [rand(rng, d0)[1] for _ in 1:3])
    end

    @testset "array shapes" begin
        dm = ComplexVisLikelihood(reshape(μ, 20, 5), reshape(w, 20, 5))
        @test logdensityof(dm, reshape(x, 20, 5)) ≈ logdensityof(d, x)
        @test size(rand(rng, dm)) == (20, 5)
    end

    @testset "sampling" begin
        s = stack(rand(rng, d) for _ in 1:50_000)
        @test within(vec(mean(s; dims = 2)), μ, sqrt.(2 ./ w), size(s, 2))
        @test vec(var(real.(s); dims = 2)) ≈ inv.(w) rtol = 3.0e-2
        @test vec(mean(abs2.(s .- μ); dims = 2)) ≈ var(d) rtol = 3.0e-2
    end

    @testset "errors" begin
        @test_throws "finite and non-negative" ComplexVisLikelihood(μ, -w)
        @test_throws DimensionMismatch ComplexVisLikelihood(μ, w[1:10])
        @test_throws DimensionMismatch logdensityof(d, x[1:10])
    end

    @testset "gradient" begin
        f(p) = logdensityof(ComplexVisLikelihood(complex.(p[1:n], p[(n + 1):end]), w, d.lognorm), x)
        check_gradient(f, vcat(real.(μ), imag.(μ)) .+ 0.1)
    end
end
