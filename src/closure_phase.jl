export NormalClosurePhaseLikelihood, VonMisesClosurePhaseLikelihood

struct NormalClosurePhaseLikelihood{V1,V2,W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end


Dists.insupport(d::NormalClosurePhaseLikelihood, x) = true

# function NormalClosurePhaseLikelihood(μ::AbstractVector{<:Complex}, σ::AbstractVector, design::AbstractMatrix)
#     Σ = PDmat(σ'*design*σ)
#     μcp = design*angle.(μ)
#     lognorm = _closurephasenorm(μcp, Σ)
#     return NormalClosurePhaseLikelihood1(μcp, Σ, lognorm)
# end

function NormalClosurePhaseLikelihood(μ::AbstractVector, Σ::AbstractVector)
    lognorm = _closurephasenorm(μ, Σ)
    return NormalClosurePhaseLikelihood(μ, Σ, lognorm)
end

function NormalClosurePhaseLikelihood(μ::AbstractVector, Σ::Diagonal)
    return NormalClosurePhaseLikelihood(μ, diag(Σ))
end

function NormalClosurePhaseLikelihood(μ::AbstractVector, Σ::AbstractMatrix)
    Σpd = PDMat(Σ)
    lognorm = _closurephasenorm(μ, Σpd)
    return NormalClosurePhaseLikelihood(μ, Σpd, lognorm)
end

function _closurephasenorm(μ, Σ::AbstractVector)
    lw =  sum(log, Σ)
    return (-lw - length(μ)*log2π)/2
end

function _closurephasenorm(μ, Σ::PDMat)
    lw = logdet(Σ)
    return (-lw - length(μ)*log2π)/2
end

function unnormed_logpdf(d::NormalClosurePhaseLikelihood{V,P}, x) where {V, P<:PDMat}
    z = invquad(d.Σ, x .- d.μ)
    return -z/2
end


function unnormed_logpdf(d::NormalClosurePhaseLikelihood{V,P}, x) where {V, P<:AbstractVector}
    _logdensity_def_μΣ(d.μ, d.Σ, x)
end



struct VonMisesClosurePhaseLikelihood{V1, V2<:AbstractVector, W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end


function VonMisesClosurePhaseLikelihood(μ::AbstractVector, Σ::AbstractVector)
    lognorm = _vonmisesclosurephasenorm(μ, Σ)
    return VonMisesClosurePhaseLikelihood(μ, Σ, lognorm)
end



function _vonmisesclosurephasenorm(μ, Σ)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    return -n*log2π - sum(x->log(besseli0x(inv(x))), Σ)
end

function unnormed_logpdf(d::VonMisesClosurePhaseLikelihood, x)
    s = zero(eltype(x))
    @simd for i in eachindex(d.μ, d.Σ)
        s += (cos(x[i] - d.μ[i]) - 1)*inv(d.Σ[i])
    end
    return s
end

function ChainRulesCore.rrule(::typeof(unnormed_logpdf), d::VonMisesClosurePhaseLikelihood, x)
    s = zero(eltype(x))
    dx = zero(x)
    dμ = zero(d.μ)
    dΣ = zero(d.Σ)

    @simd for i in eachindex(d.μ, d.Σ)
        Σinv = inv(d.Σ[i])
        si,ci = sincos(x[i] - d.μ[i])
        s += (ci-1)*Σinv
        dμ[i] = si*Σinv
        dx[i] = -si*Σinv
        dΣ[i] = (ci - 1)/Σinv
    end

    function _unormed_logpdf_vonmisescp(Δ)
        dμ .= Δ.*dμ
        dx .= Δ.*dx
        dΣ .= Δ.*dΣ

        return NoTangent(), Tangent{typeof(d)}(μ = dμ, Σ = dΣ, lognorm=ZeroTangent()), dx
    end
    return s/2, _unormed_logpdf_vonmisescp
end

Dists.insupport(d::VonMisesClosurePhaseLikelihood, x) = true
