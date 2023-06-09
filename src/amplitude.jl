export AmplitudeLikelihood, RiceAmplitudeLikelihood

struct AmplitudeLikelihood{V1,V2<:Union{AbstractVector, AbstractPDMat},W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

Base.length(d::AmplitudeLikelihood) = length(d.μ)
Base.eltype(d::AmplitudeLikelihood) = promote_type(eltype(d.μ), eltype(d.Σ))
Dists.insupport(::AmplitudeLikelihood, x) = true

"""
    AmplitudeLikelihood(μ, Σ::Union{AbstractVector, AbstractMatrix})

Forms the likelihood for amplitudes from the mean vector `μ` and
the covariance matrix `Σ`. If Σ is vector or a diagonal
matrix then we assume that the argument is the diagonal covariance matrix.
If Σ is a full matrix then we assume that a dense covariance was passed

# Notes

We do no processing to the data, i.e. the mean μ is not-debiased anywhere.

# Warning

This likelihood will be significantly biased from the true Rice distribution
for data points with SNR = μ/Σ < 2. If this matter significantly for you, we recommend
that you consider fitting pure complex visibilities instead.
"""
function AmplitudeLikelihood(μ::AbstractVector, Σ::AbstractVector)
    lognorm = _gaussnorm(μ, Σ)
    return AmplitudeLikelihood(μ, Σ, lognorm)
end

function AmplitudeLikelihood(μ::AbstractVector, Σ::Diagonal)
    AmplitudeLikelihood(μ, diag(Σ))
end


function AmplitudeLikelihood(μ::AbstractVector, Σ::AbstractMatrix)
    Σpd = PDMat(Σ)
    return AmplitudeLikelihood(μ, Σpd)
end


function AmplitudeLikelihood(μ::AbstractVector, Σ::AbstractPDMat)
    lognorm = _gaussnorm(μ, Σ)
    return AmplitudeLikelihood(μ, Σ, lognorm)
end




function unnormed_logpdf(d::AmplitudeLikelihood, x::AbstractVector)
    return _unnormed_logpdf_μΣ(d.μ, d.Σ, x)
end

function unnormed_logpdf(d::AmplitudeLikelihood{V,P}, x::AbstractVector) where {V, P<:AbstractPDMat}
    return _amp_logpdf_full(d.μ, d.Σ, x)
end

function _amp_logpdf_full(μ, Σ, x)
    dθ = x .- μ
    z = _chi2(dθ, Σ)
    return -z
end

function Distributions._rand!(rng::Random.AbstractRNG, d::AmplitudeLikelihood{<:AbstractVector, <:AbstractVector}, x::AbstractVector)
    randn!(rng, x)
    x .= x.*sqrt.(d.Σ) .+ d.μ
    return x
end

function Distributions._rand!(rng::Random.AbstractRNG, d::AmplitudeLikelihood{<:AbstractVector, <:AbstractPDMat}, x::AbstractVector)
    randn!(rng, x)
    PDMats.unwhiten!(d.Σ, x)
    x .+= d.μ
    return x
end

struct RiceAmplitudeLikelihood{V1,V2<:AbstractVector,W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

Base.length(d::RiceAmplitudeLikelihood) = length(d.μ)
Base.eltype(d::RiceAmplitudeLikelihood) = promote_type(eltype(d.μ), eltype(d.Σ))
Dists.insupport(::RiceAmplitudeLikelihood, x) = true

"""
    RiceAmplitudeLikelihood(μ, Σ::Union{AbstractVector, Diagonal})

Forms the likelihood for amplitudes from the mean vector `μ` and
the diagonal covariance matrix `Σ`. `Σ` can either be a Diagonal matrix or
a vector whose entries are the variance for each data point.

# Notes
This is the correct likelihood distribution for visibility amplitudes, but it is slower
than the Gaussian approximation `AmplitudeLikelihood`. Additionally, i
"""
function RiceAmplitudeLikelihood(μ::AbstractVector, Σ::AbstractVector)
    lognorm = -sum(log, Σ)
    return RiceAmplitudeLikelihood(μ, Σ, lognorm)
end

function RiceAmplitudeLikelihood(μ::AbstractVector, Σ::Diagonal)
    return RiceAmplitudeLikelihood(μ, diag(Σ))
end

_L12(x) = exp(x/2)*((1-x)*besseli0(-x/2) - x*besseli1(-x/2))

function unnormed_logpdf(d::RiceAmplitudeLikelihood, x::AbstractVector)
    return sum(zip(d.μ, d.Σ, x)) do (μ, Σ, xx)
        # we use besseli0x for numerical stability for high SNR points
        return -(xx^2 + μ^2)*inv(2*Σ) + log(xx*besseli0x(xx*μ/Σ)) + xx*μ/Σ
    end
end

function ChainRulesCore.rrule(::typeof(unnormed_logpdf), d::RiceAmplitudeLikelihood, x::AbstractVector)
    ℓ =
    function _ricelklhd_logpdf_pullback(Δ)

    end

end

Dists.mean(d::RiceAmplitudeLikelihood) = sqrt.(d.Σ .* π/2).*_L12.(-d.μ.^2 .* inv.(2 .* d.Σ))
function Dists.var(d::RiceAmplitudeLikelihood)
    T = eltype(d)
    return 2*d.Σ .+ d.μ.^2 - (T(π).*d.Σ/2).*_L12.(-d.μ.^2 .* inv.(2*d.Σ)).^2
end


function Distributions._rand!(rng::Random.AbstractRNG, d::RiceAmplitudeLikelihood{<:AbstractVector, <:AbstractVector}, x::AbstractVector)
    randn!(rng, x)
    # sample from a normal gaussian and abs.
    x .= abs.(complex.(x.*sqrt.(d.Σ) .+ d.μ, randn.(rng).*sqrt.(d.Σ)))
    return x
end
