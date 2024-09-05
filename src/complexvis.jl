export ComplexVisLikelihood

"""
    ComplexVisLikelihood(μ::AbstractVector{<:Complex}, Σ::AbstractVector{<:Real})

Creates the complex visibility likelihood distribution which is a diagonal complex Gaussian
with strictly real covariance matrix.

## Paramters
- `μ`: The mean complex visibility, which is usually computed from some VLBI model
- `Σ`: The measurement covariance matrix, which is usually computed directly from the data.
       Note that `Σ` must be a real element vector and is interpreted at the diagonal of the
       covariance matrix.
"""
struct ComplexVisLikelihood{V1<:AbstractArray{<:Complex},V2,W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

Base.length(d::ComplexVisLikelihood) = length(d.μ)
Base.eltype(d::ComplexVisLikelihood) = eltype(d.μ)
Dists.insupport(d::ComplexVisLikelihood, x) = true
Dists.mean(d::ComplexVisLikelihood) = d.μ
Dists.var(d::ComplexVisLikelihood) = d.Σ
Dists.cov(d::ComplexVisLikelihood) = Diagonal(Dists.var(d))


function ComplexVisLikelihood(μ::AbstractVector{<:Complex}, Σ::AbstractVector{<:Real})
    lognorm = _cvisnorm(μ, Σ)
    return ComplexVisLikelihood(μ, Σ, lognorm)
end

function _cvisnorm(μ, Σ)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    # 2x the amplitude version because of real and imag components
    return 2*_gaussnorm(μ, Σ)
end


function unnormed_logpdf(d::ComplexVisLikelihood, x::AbstractVector{<:Complex})
    return _unnormed_logpdf_μΣ(d.μ, d.Σ, x)
end

function Dists._rand!(rng::Random.AbstractRNG, d::ComplexVisLikelihood, x::AbstractArray{<:Complex})
    x .= randn!(rng, x).*sqrt.(d.Σ)
    x .+= d.μ
    return x
end

# function Dists.rand!(rng::Random.AbstractRNG, d::ComplexVisLikelihood, x::AbstractVector, dims::Dims)
#     x .= randn(rng, eltype(d)).*sqrt.(d.Σ) .+ d.μ
# end
