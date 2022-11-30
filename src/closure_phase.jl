export ClosurePhaseLikelihood

struct ClosurePhaseLikelihood{V1,V2,W} <: AbstractVLBIDistributions
    μ::V1
    Σ::V2
    lognorm::W
end

Base.length(d::ClosurePhaseLikelihood) = length(d.μ)
Base.eltype(d::ClosurePhaseLikelihood) = promote_type(eltype(d.μ), eltype(d.Σ))
Dists.insupport(d::ClosurePhaseLikelihood, x) = true


function ClosurePhaseLikelihood(μ::AbstractVector, Σ::AbstractVector)
    lognorm = _closurephasenorm(μ, Σ)
    return ClosurePhaseLikelihood(μ, Σ, lognorm)
end

function ClosurePhaseLikelihood(μ::AbstractVector, Σ::Diagonal)
    return ClosurePhaseLikelihood(μ, diag(Σ))
end

function ClosurePhaseLikelihood(μ::AbstractVector, Σ::AbstractMatrix)
    Σpd = PDMat(Σ)
    lognorm = _closurephasenorm(μ, Σpd)
    return ClosurePhaseLikelihood(μ, Σpd, lognorm)
end

function _closurephasenorm(μ, Σ::AbstractVector)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    return -n*log2π - sum(x->log(besseli0x(inv(x))), Σ)
end

function ChainRulesCore.rrule(::typeof(_closurephasenorm), μ, Σ::AbstractVector)
    v =zero(eltype(Σ))
    dΣ = zero(Σ)
    for i in eachindex(Σ)
        κ = inv(Σ[i])
        i0 = besseli0x(κ)
        i1 = besseli1x(κ)
        v += log(i0)
        dΣ[i] = Σ[i] + i1/i0
    end
    function _closurephasenorm_pullback(Δ)
        dΣ .= -Δ.*dΣ
        return NoTangent(), ZeroTangent(), dΣ
    end
    return v, _closurephasenorm_pullback
end

function _closurephasenorm(μ, Σ::PDMat)
    lw = logdet(Σ)
    return (-lw - length(μ)*log2π)/2
end

# We mark the norms as non-differentiable. Why? Because they are wrong anyways!
ChainRulesCore.@non_differentiable _closurephasenorm(μ, Σ::PDMat)

function unnormed_logpdf(d::ClosurePhaseLikelihood{V,P}, x) where {V, P<:PDMat}
    dθ = cis.(x) .- cis.(d.μ)
    z = invquad(d.Σ, dθ)
    return -z/2
end


function unnormed_logpdf(d::ClosurePhaseLikelihood{V,P}, x) where {V, P<:AbstractVector}
    s = zero(eltype(x))
    @simd for i in eachindex(d.μ, d.Σ)
        s += (cos(x[i] - d.μ[i]) - 1)*inv(d.Σ[i])
    end
    return s
end


function ChainRulesCore.rrule(::typeof(unnormed_logpdf), d::ClosurePhaseLikelihood{V,<:AbstractVector}, x) where {V}
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

    function _unnormed_logpdf_vonmisescp(Δ)
        dμ .= Δ.*dμ
        dx .= Δ.*dx
        dΣ .= Δ.*dΣ

        return NoTangent(), Tangent{typeof(d)}(μ = dμ, Σ = dΣ, lognorm=ZeroTangent()), dx
    end
    return s/2, _unnormed_logpdf_vonmisescp
end
