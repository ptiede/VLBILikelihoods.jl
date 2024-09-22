using EnzymeTestUtils
using LinearAlgebra
using SparseArrays
using Enzyme

@testset "Enzyme CholeskyFactor" begin
    N = 50
    σ = sprand(N,N, 0.1)
    Σ = 0.5.*(σ + σ') + 5 .* Diagonal(ones(N))
    C = VLBILikelihoods.CholeskyFactor(Σ)

    v = randn(N)

    for Tret in (Duplicated, BatchDuplicated), Tv in (Duplicated, BatchDuplicated)
        are_activities_compatible(Tret, Tv) || continue
        test_reverse(\, Tret, (C, Const), (v, Tv))
    end
end