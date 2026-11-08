@testset "Coherency Likelihood" begin
    rng = Random.Xoshiro(2)
    n = 10
    comps(f) = ntuple(_ -> f(), 4)
    μc = comps(() -> complex.(rand(rng, n), rand(rng, n)))
    xc = comps(() -> complex.(rand(rng, n), rand(rng, n)))
    wc = comps(() -> inv.(rand(rng, n) .+ 3.0))

    μ = StructVector(SMatrix{2, 2}.(μc...))
    x = StructVector(SMatrix{2, 2}.(xc...))
    w = StructVector(SMatrix{2, 2}.(wc...))
    d = CoherencyLikelihood(μ, w)

    @test eltype(d) == eltype(μ)
    @test length(d) == length(μ)

    expected = sum(1:4) do k
        logpdf(MvNormal(real.(μc[k]), Diagonal(inv.(wc[k]))), real.(xc[k])) +
            logpdf(MvNormal(imag.(μc[k]), Diagonal(inv.(wc[k]))), imag.(xc[k]))
    end
    @test logdensityof(d, x) ≈ expected
    @inferred Float64 logdensityof(d, x)

    @testset "storage layouts" begin
        darr = CoherencyLikelihood(Array(μ), Array(w))
        @test darr.lognorm ≈ d.lognorm
        @test logdensityof(darr, Array(x)) ≈ logdensityof(d, x)
        dense(c) = reshape(stack(c), n, 2, 2)
        ddense = CoherencyLikelihood(dense(μc), dense(wc))
        @test logdensityof(ddense, dense(xc)) ≈ logdensityof(d, x)
        @test_throws DimensionMismatch CoherencyLikelihood(rand(ComplexF64, n, 3), rand(n, 3))
    end

    @testset "missing hands" begin
        w0 = StructVector(SMatrix{2, 2}.(wc[1], zero(wc[2]), wc[3], wc[4]))
        x0 = StructVector(SMatrix{2, 2}.(xc[1], zero(xc[2]), xc[3], xc[4]))
        d0 = CoherencyLikelihood(μ, w0)
        @test logdensityof(d0, x0) ≈ expected -
            logpdf(MvNormal(real.(μc[2]), Diagonal(inv.(wc[2]))), real.(xc[2])) -
            logpdf(MvNormal(imag.(μc[2]), Diagonal(inv.(wc[2]))), imag.(xc[2]))
    end

    @testset "sampling" begin
        s = [rand(rng, d) for _ in 1:50_000]
        for k in 1:4
            sk = stack(StructArrays.component(si, k) for si in s)
            @test within(vec(mean(sk; dims = 2)), μc[k], sqrt.(2 ./ wc[k]), size(sk, 2))
            @test vec(var(real.(sk); dims = 2)) ≈ inv.(wc[k]) rtol = 3.0e-2
        end
    end
end
