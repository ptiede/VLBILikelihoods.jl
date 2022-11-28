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
    logw = -4*n*log2π - sum(sum(x->log.(x), Σ))
    return logw
end

function ChainRulesCore.rrule(::typeof(_coherencynorm), μ, Σ)
    s = _coherencynorm(μ, Σ)
    function _coherencynorm_pullback(Δ)
        dΣ = -Δ.*map(x->inv.(x), Σ)
        return NoTangent(), ZeroTangent(), dΣ
    end
    return s, _coherencynorm_pullback
end

function unnormed_logpdf(d::CoherencyLikelihood, x::AbstractVector{<:StaticArraysCore.SMatrix{2,2}})
    s = _unnormed_logpdf_μΣ(d.μ, d.Σ, x)
    return s[1] + s[2] + s[3] + s[4]
end

# function ChainRulesCore.rrule(::typeof(unnormed_logpdf), d::CoherencyLikelihood, x::AbstractVector{<:StaticArraysCore.SMatrix{2,2}})
#     s, ds = ChainRulesCore.rrule(_unnormed_logpdf_μΣ, d.μ, d.Σ, x)
#     function _unnormed_logpdf_coherency_pullback(Δ)
#         gd = ds(Δ)
#         return gd
#     end
#     return sum(s)
# end

# function _unnormed_logpdf_coh(μ, Σ, x)
#     s = zero(eltype(eltype(Σ)))
#     @inbounds @simd for i in eachindex(μ, Σ, x)
#         s += abs2(x[i][1] - μ[i][1])*inv(Σ[i][1])
#         s += abs2(x[i][2] - μ[i][2])*inv(Σ[i][2])
#         s += abs2(x[i][3] - μ[i][3])*inv(Σ[i][3])
#         s += abs2(x[i][4] - μ[i][4])*inv(Σ[i][4])
#     end
#     return -s/2
# end

# function ChainRulesCore.rrule(::typeof(unnormed_logpdf_coh), μ, Σ, x)
#     s = zero(eltype(Σ))
#     dx = zero(x)
#     dμ = zero(μ)
#     dΣ = zero(Σ)
#     @simd for i in eachindex(μ, Σ)
#         Δx = abs(x[i] - μ[i])
#         Σinv = inv(Σ)
#         s += -Δx^2*Σinv
#         dμ[i] = dx[i] = -Δx*Σinv
#         dΣ[i] = Δx^2*Σinv^2/2
#     end

#     function _unnormed_logpdf_coh(Δ)
#         dμ .= -Δ.*dμ
#         dx .= Δ.*dx
#         dΣ .= Δ.*dΣ
#         return NoTangent(), dμ, dΣ, dx
#     end

#     return s/2, _unnormed_logpdf_coh
# end
