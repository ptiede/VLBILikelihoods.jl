export CoherencyLikelihood


"""
    ComplexVisLikelihood(μ::AbstractVector{<:StaticMatrix{2,2}}, Σ::AbstractVector{<:StaticMatrix{2,2, <:Real}})

Creates the coherency matrix likelihood distribution which is a Gaussian over the space of
2×2 complex visibilities.

## Paramters
- `μ`: The mean coherency matrix stored as a vector of 2x2 static matrices, which is usually computed from some VLBI model
- `Σ`: The measurement covariance matrix, which is usually computed directly from the data.
       Note that `Σ` must vector of 2x2 real static matrices.

# Notes

You will get the best performance is all the vectors are given as StructVectors{<:StaticMatrices{2,2}}, especially
when computing gradients with an AD engine like Zygote.
"""
struct CoherencyLikelihood{V1,V2,W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

Base.length(d::CoherencyLikelihood) = length(d.μ)
Base.eltype(d::CoherencyLikelihood) = eltype(d.μ)
Dists.insupport(d::CoherencyLikelihood, x) = true

function CoherencyLikelihood(μ::AbstractVector{<:SA.StaticMatrix{2,2}}, Σ::AbstractVector{<:SA.StaticMatrix{2,2}})
    return CoherencyLikelihood(StructArray(μ), StructArray(Σ))
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



function unnormed_logpdf(d::CoherencyLikelihood{<:StructVector{<:SA.StaticMatrix{2,2}}, <:StructVector{<:SA.StaticMatrix{2,2}}},
                         x::StructVector{<:SA.StaticMatrix{2,2}})
    μs = values(StructArrays.components(d.μ))
    Σs = values(StructArrays.components(d.Σ))
    xs = values(StructArrays.components(x))
    s = sum(_unnormed_logpdf_μΣ.(μs, Σs, xs))
    return s
end

function unnormed_logpdf(d::CoherencyLikelihood{<:StructVector{<:SA.StaticMatrix{2,2}}, <:StructVector{<:SA.StaticMatrix{2,2}}},
                         x::AbstractVector{<:SA.StaticMatrix{2,2}})
    return unnormed_logpdf(d, StructVector(x))
end
