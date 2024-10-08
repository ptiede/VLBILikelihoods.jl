export CoherencyLikelihood


"""
    ComplexVisLikelihood(μ::AbstractVector{<:StaticMatrix{2,2}}, Σ::AbstractVector{<:StaticMatrix{2,2, <:Real}})

Creates the coherency matrix likelihood distribution which is a Gaussian over the space of
2×2 complex visibilities.

## Paramters
- `μ`: The mean coherency matrix stored as a vector of 2x2 static matrices, which is usually computed from some VLBI model
- `Σ`: The measurement covariance matrix, which is usually computed directly from the data.
       Note that `Σ` must vector of 2x2 real static matrices.

!!! notes
    You will get the best performance is all the vectors are given as StructVectors{<:StaticMatrices{2,2}}.
"""
struct CoherencyLikelihood{V1,V2,W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

Base.length(d::CoherencyLikelihood) = length(d.μ)
Base.eltype(d::CoherencyLikelihood) = eltype(d.μ)
Dists.insupport(d::CoherencyLikelihood, x) = true
Dists.mean(d::CoherencyLikelihood) = d.μ
Dists.var(d::CoherencyLikelihood) = d.Σ

function CoherencyLikelihood(μ::AbstractVector{<:SA.StaticMatrix{2,2}}, Σ::AbstractVector{<:SA.StaticMatrix{2,2}})
    return CoherencyLikelihood(StructArray(μ), StructArray(Σ))
end

function CoherencyLikelihood(μ::UnstructuredMap{<:SA.StaticMatrix{2,2}}, Σ::AbstractVector{<:SA.StaticMatrix{2,2}})
    return CoherencyLikelihood(parent(μ), Σ)
end

function CoherencyLikelihood(μ::AbstractVector{<:SA.StaticMatrix{2,2}}, Σ::UnstructuredMap{<:SA.StaticMatrix{2,2}})
    return CoherencyLikelihood(μ, parent(Σ))
end

function CoherencyLikelihood(μ::UnstructuredMap{<:SA.StaticMatrix{2,2}}, Σ::UnstructuredMap{<:SA.StaticMatrix{2,2}})
    return CoherencyLikelihood(parent(μ), parent(Σ))
end




function CoherencyLikelihood(μ::StructVector{<:SA.StaticMatrix{2,2}}, Σ::StructVector{<:SA.StaticMatrix{2,2}})
    lognorm = _coherencynorm(μ, Σ)
    return CoherencyLikelihood(μ, Σ, lognorm)
end

function _coherencynorm(μ::StructVector, Σ::StructVector)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    μs = StructArrays.components(μ)
    Σs = StructArrays.components(Σ)
    sum(_cvisnorm.(μs, Σs))
end

unnormed_logpdf(d::CoherencyLikelihood, x::UnstructuredMap) = unnormed_logpdf(d, baseimage(x))
function unnormed_logpdf(d::CoherencyLikelihood{<:StructVector{<:SA.StaticMatrix{2,2}}, <:StructVector{<:SA.StaticMatrix{2,2}}},
                         x::StructVector{<:SA.StaticMatrix{2,2}})
    μs = values(StructArrays.components(d.μ))
    Σs = values(StructArrays.components(d.Σ))
    xs = values(StructArrays.components(x))
    s1 = _unnormed_logpdf_μΣ(μs[1], Σs[1], xs[1])
    s2 = _unnormed_logpdf_μΣ(μs[2], Σs[2], xs[2])
    s3 = _unnormed_logpdf_μΣ(μs[3], Σs[3], xs[3])
    s4 = _unnormed_logpdf_μΣ(μs[4], Σs[4], xs[4])
    return s1+s2+s3+s4
end

function unnormed_logpdf(d::CoherencyLikelihood{<:StructVector{<:SA.StaticMatrix{2,2}}, <:StructVector{<:SA.StaticMatrix{2,2}}},
                         x::AbstractVector{<:SA.StaticMatrix{2,2}})
    return unnormed_logpdf(d, StructVector(x))
end

function Dists._rand!(rng::Random.AbstractRNG, d::CoherencyLikelihood, x::StructVector{<:SA.StaticMatrix{2,2}})
    μ = d.μ
    Σ = d.Σ
    x.:1 .= randn!(rng, x.:1).*sqrt.(Σ.:1)
    x.:2 .= randn!(rng, x.:2).*sqrt.(Σ.:2)
    x.:3 .= randn!(rng, x.:3).*sqrt.(Σ.:3)
    x.:4 .= randn!(rng, x.:4).*sqrt.(Σ.:4)
    x .+= μ
    return x
end

function Dists._rand!(rng::Random.AbstractRNG, d::CoherencyLikelihood, x::StructArray{<:SA.StaticMatrix{2,2}, N}) where {N}
    Ind = CartesianIndices(size(x)[2:N])
    for I in Ind
        Dists._rand!(rng, d, @view(x[:, I]))
    end
    return x
end



Dists.rand(rng::Random.AbstractRNG, d::CoherencyLikelihood) = Dists._rand!(rng, d, similar(d.μ))
Dists.rand(rng::Random.AbstractRNG, d::CoherencyLikelihood, dims::Int...) = Dists._rand!(rng, d, similar(d.μ, eltype(d.μ), size(d.μ)..., dims...))
