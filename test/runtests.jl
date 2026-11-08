using VLBILikelihoods
using Test
using Distributions
using StaticArrays
using LinearAlgebra
using FiniteDifferences
using StructArrays
using SparseArrays
using Serialization
using Enzyme
using Random

# Enzyme's gradient of the real function `f` at `x` against central finite differences.
function check_gradient(f, x; rtol = 1.0e-6)
    # Runtime activity as in Comrade: the likelihood holds active means next to constant weights.
    ge = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(f), x)[1]
    gf = FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]
    @test ge ≈ gf rtol = rtol
end

# Each mean within `k` standard errors of `μ`.
within(m, μ, σ, n; k = 5) = all(abs.(m .- μ) .<= k .* σ ./ sqrt(n))

randpd(n) = (m = sprand(n, n, 0.05); 0.5 .* (m + m') + 5 .* Diagonal(ones(n)))

@testset "VLBILikelihoods.jl" begin
    @testset "CholeskyFactor" begin
        Σ = randpd(50)
        c = CholeskyFactor(Σ)
        @test parent(c) === Σ
        @test size(Σ) == size(c)
        @test getindex(Σ, 1, 1) == getindex(c, 1, 1)
        @test cholesky(c) === c
        @test adjoint(c) isa CholeskyFactor

        path = tempname()
        serialize(path, c)
        c2 = deserialize(path)
        rm(path)
        x = rand(size(c2, 1))
        @test c2.cov == c.cov
        @test c2 \ x == c \ x
    end

    include(joinpath(@__DIR__, "amplitude.jl"))
    include(joinpath(@__DIR__, "closure_phase.jl"))
    include(joinpath(@__DIR__, "complex_vis.jl"))
    include(joinpath(@__DIR__, "coherency.jl"))
    include(joinpath(@__DIR__, "rules.jl"))
end
