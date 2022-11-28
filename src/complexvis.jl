export ComplexVisLikelihood

struct ComplexVisLikelihood{V1<:AbstractArray{<:Complex},V2,W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

Base.length(d::ComplexVisLikelihood) = length(d.μ)
Base.eltype(d::ComplexVisLikelihood) = eltype(d.μ)
Dists.insupport(d::ComplexVisLikelihood, x) = true

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

function ChainRulesCore.rrule(::Type{<:ComplexVisLikelihood}, μ::AbstractVector, Σ::AbstractVector)
    lognorm = AmplitudeLikelihood(μ, Σ, _complexvisnorm(μ, Σ))
    function _ComplexVisLikelihood_pullback(Δ)
        Δμ = Δ.μ
        ΔΣ = Δ.Σ .- Δ.lognorm.*inv.(Σ)
        return NoTangent(), Δμ, ΔΣ
    end
    return lognorm, _ComplexVisLikelihood_pullback
end

function unnormed_logpdf(d::ComplexVisLikelihood, x::AbstractVector{<:Complex})
    return _unnormed_logpdf_μΣ(d.μ, d.Σ, x)
end
