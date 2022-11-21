export AmplitudeLikelihood

struct AmplitudeLikelihood{V1,V2,W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

function AmplitudeLikelihood(μ::AbstractVector, Σ::AbstractVector)
    lognorm = _amplitudenorm(μ, Σ)
    return AmplitudeLikelihood(μ, Σ, lognorm)
end

function AmplitudeLikelihood(μ::AbstractVector, Σ::Diagonal)
    AmplitudeLikelihood(μ, diag(Σ))
end

function ChainRulesCore.rrule(::Type{<:AmplitudeLikelihood}, μ::AbstractVector, Σ::AbstractVector)
    lognorm = AmplitudeLikelihood(μ, Σ, _amplitudenorm(μ, Σ))
    function _AmplitudeLikelihood_pullback(Δ)
        Δμ = Δ.μ
        ΔΣ = Δ.Σ .- Δ.lognorm.*inv.(Σ)/2
        return NoTangent(), Δμ, ΔΣ
    end
    return lognorm, _AmplitudeLikelihood_pullback
end

Base.length(d::AmplitudeLikelihood) = length(d.μ)
Base.eltype(d::AmplitudeLikelihood) = promote_type(eltype(d.μ), eltype(d.Σ))

function unnormed_logpdf(d::AmplitudeLikelihood, x::AbstractArray)
    return _logdensity_def_μΣ(d.μ, d.Σ, x)
end

Dists.insupport(::AmplitudeLikelihood, x) = true

function _amplitudenorm(μ, Σ)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    logw = -n/2*log2π - sum(log, Σ)/2
    return logw
end


function _logdensity_def_μΣ(μ, Σ, x)
    s = zero(eltype(x))
    @simd for i in eachindex(μ, Σ)
        s += -(x[i] - μ[i])^2*inv(Σ[i])
    end
    return s/2
end

function ChainRulesCore.rrule(::typeof(_logdensity_def_μΣ), μ, Σ, x)
    s = zero(eltype(x))
    dx = zero(x)
    dμ = zero(μ)
    dΣ = zero(Σ)
    @simd for i in eachindex(μ, Σ)
        Δx = (x[i] - μ[i])
        Σinv = inv(Σ)
        s += -Δx^2*Σinv
        dμ[i] = dx[i] = -Δx*Σinv
        dΣ[i] = Δx^2*Σinv^2/2
    end

    function _logdensity_def_μΣ_pullback(Δ)
        dμ .= -Δ.*dμ
        dx .= Δ.*dx
        dΣ .= Δ.*dΣ
        return NoTangent(), dμ, dΣ, dx
    end

    return s/2, _logdensity_def_μΣ_pullback
end
