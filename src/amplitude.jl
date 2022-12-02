export AmplitudeLikelihood

struct AmplitudeLikelihood{V1,V2<:Union{AbstractVector, AbstractPDMat},W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

Base.length(d::AmplitudeLikelihood) = length(d.μ)
Base.eltype(d::AmplitudeLikelihood) = promote_type(eltype(d.μ), eltype(d.Σ))
Dists.insupport(::AmplitudeLikelihood, x) = true

"""
    AmplitudeLikelihood(μ, Σ::Union{AbstractVector, Diagonal})

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

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{<:Union{AmplitudeLikelihood, ClosurePhaseLikelihood}}, μ::AbstractVector, Σ::AbstractMatrix)
    Σpd = PDMat(Σ)
    d = T(μ, Σpd)

    # get the normalization from the rrule
    function _AmplitudeNormal_pullback(Δ)
        Δlg = last(rrule_via_ad(config, _gaussnorm, μ, Σpd)[2](Δ.lognorm))
        Δμ = Δ.μ
        ΔΣ = Δ.Σ + Δlg
        return NoTangent(), Δμ, ΔΣ
    end
    return d, _AmplitudeNormal_pullback
end

function AmplitudeLikelihood(μ::AbstractVector, Σ::PDMat)
    lognorm = _gaussnorm(μ, Σ)
    return AmplitudeLikelihood(μ, Σ, lognorm)
end


function ChainRulesCore.rrule(::Type{<:AmplitudeLikelihood}, μ::AbstractVector, Σ::AbstractVector)
    lognorm = AmplitudeLikelihood(μ, Σ, _gaussnorm(μ, Σ))
    function _AmplitudeLikelihood_pullback(Δ)
        d = unthunk(Δ)
        Δμ = @thunk(d.μ)
        ΔΣ = @thunk(d.Σ .- d.lognorm.*inv.(Σ)/2)
        return NoTangent(), Δμ, ΔΣ
    end
    return lognorm, _AmplitudeLikelihood_pullback
end


function unnormed_logpdf(d::AmplitudeLikelihood, x::AbstractVector)
    return _unnormed_logpdf_μΣ(d.μ, d.Σ, x)
end

function unnormed_logpdf(d::AmplitudeLikelihood{V,P}, x::AbstractVector) where {V, P<:PDMat}
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

function Distributions._rand!(rng::Random.AbstractRNG, d::AmplitudeLikelihood{<:AbstractVector, <:PDMat}, x::AbstractVector)
    randn!(rng, x)
    PDMats.unwhiten!(d.Σ, x)
    x .+= d.μ
    return x
end
