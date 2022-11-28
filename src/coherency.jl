export CoherencyLikelihood

struct CoherencyLikelihood{V1,V2,W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

Base.length(d::CoherencyLikelihood) = length(d.μ)
Base.eltype(d::CoherencyLikelihood) = eltype(d.μ)
Dists.insupport(d::CoherencyLikelihood, x) = true

function CoherencyLikelihood(μ::AbstractVector{<:SMatrix{2,2}}, Σ::AbstractVector{<:SMatrix{2,2,<:Real}})
    lognorm = _coherencynorm(μ, Σ)
    return CoherencyLikelihood(μ, Σ, lognorm)
end

function _coherencynorm(μ, Σ)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    # 2x the amplitude version because of real and imag components
    logw = -4*n*log2π - sum(sum(x->log.(x), Σ))
    return logw
end

function unnormed_logpdf(d::CoherencyLikelihood, x::AbstractVector{<:StaticArraysCore.SMatrix{2,2}})
    μ = d.μ
    Σ = d.Σ
    s = zero(eltype(eltype(d.Σ)))
    @inbounds @simd for i in eachindex(μ, Σ, x)
        s += abs2(x[i][1] - μ[i][1])*inv(Σ[i][1])
        s += abs2(x[i][2] - μ[i][2])*inv(Σ[i][2])
        s += abs2(x[i][3] - μ[i][3])*inv(Σ[i][3])
        s += abs2(x[i][4] - μ[i][4])*inv(Σ[i][4])
    end
    return -s/2
end
