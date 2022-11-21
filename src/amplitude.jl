export AmplitudeLikelihood

struct AmplitudeLikelihood{V1,V2,W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

Base.length(d::AmplitudeLikelihood) = length(d.μ)
Base.eltype(d::AmplitudeLikelihood) = promote_type(eltype(d.μ), eltype(d.Σ))
Dists.insupport(::AmplitudeLikelihood, x) = true


function AmplitudeLikelihood(μ::AbstractVector, Σ::AbstractVector)
    lognorm = _gaussnorm(μ, Σ)
    return AmplitudeLikelihood(μ, Σ, lognorm)
end

function AmplitudeLikelihood(μ::AbstractVector, Σ::Diagonal)
    AmplitudeLikelihood(μ, diag(Σ))
end

function ChainRulesCore.rrule(::Type{<:AmplitudeLikelihood}, μ::AbstractVector, Σ::AbstractVector)
    lognorm = AmplitudeLikelihood(μ, Σ, _gaussnorm(μ, Σ))
    function _AmplitudeLikelihood_pullback(Δ)
        Δμ = Δ.μ
        ΔΣ = Δ.Σ .- Δ.lognorm.*inv.(Σ)/2
        return NoTangent(), Δμ, ΔΣ
    end
    return lognorm, _AmplitudeLikelihood_pullback
end


function unnormed_logpdf(d::AmplitudeLikelihood, x::AbstractArray)
    return _unormed_logpdf_μΣ(d.μ, d.Σ, x)
end
