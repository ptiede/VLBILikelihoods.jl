export ClosurePhaseLikelihood

struct ClosurePhaseLikelihood{V1,V2<:Union{AbstractVector, AbstractPDMat},W} <: AbstractVLBIDistributions
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
    return ClosurePhaseLikelihood(μ, Σpd)
end

function ClosurePhaseLikelihood(μ::AbstractVector, Σ::PDMat)
    lognorm = _gaussnorm(μ, Σ)
    return ClosurePhaseLikelihood(μ, Σ, lognorm)
end

function _closurephasenorm(μ, Σ::AbstractVector)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    return -n*log2π - sum(x->log(besseli0x(inv(x))), Σ)
end

function ChainRulesCore.rrule(::typeof(_closurephasenorm), μ, Σ::AbstractVector)
    v =zero(eltype(Σ))
    n = length(Σ)
    dΣ = zero(Σ)
    for i in eachindex(Σ)
        κ = inv(Σ[i])
        i0 = besseli0x(κ)
        i1 = besseli1x(κ)
        v += log(i0)
        dΣ[i] = (i1/i0-1)*κ^2
    end
    function _closurephasenorm_pullback(Δ)
       ΔΣ = Δ.*dΣ
        return NoTangent(), ZeroTangent(), ΔΣ
    end
    return -n*log2π - v, _closurephasenorm_pullback
end

# # We mark the norms as non-differentiable. Why? Because they are wrong anyways!
# ChainRulesCore.@non_differentiable _closurephasenorm(μ, Σ::PDMat)

function unnormed_logpdf(d::ClosurePhaseLikelihood{V,P}, x) where {V, P<:PDMat}
    return _cp_logpdf_full(d.μ, d.Σ, x)
end

function _cp_logpdf_full(μ, Σ, x)
    dθ = cis.(x) .- cis.(μ)
    z = _chi2(dθ, Σ)
    return -z
end


function Distributions._rand!(rng::Random.AbstractRNG, d::ClosurePhaseLikelihood{<:AbstractVector, <:PDMat}, x::AbstractVector)
    randn!(rng, x)
    PDMats.unwhiten!(d.Σ, x)
    x .+= d.μ
    return x
end

function Distributions._rand!(rng::Random.AbstractRNG, d::ClosurePhaseLikelihood{<:AbstractVector, <:AbstractVector}, x::AbstractVector)
    randn!(rng, x)
    x .= x.*sqrt.(d.Σ) .+ d.μ
    return x
end



function unnormed_logpdf(d::ClosurePhaseLikelihood{V,P}, x) where {V, P<:AbstractVector}
    return _cp_logpdf(d.μ, d.Σ, x)
end

function _cp_logpdf(μ, Σ, x)
    s = zero(eltype(x))
    @simd for i in eachindex(μ, Σ)
        s += (cos(x[i] - μ[i]) - 1)*inv(Σ[i])
    end
    return s
end


function ChainRulesCore.rrule(::typeof(_cp_logpdf), μ, Σ, x)
    s = _cp_logpdf(μ, Σ, x)

    function _cp_logpdf_pullback(Δ)
        Σinv = inv.(Σ)
        ss = sin.(x .- μ)
        dμ = @thunk(Δ.*ss.*Σinv)
        dx = @thunk(-Δ.*ss.*Σinv)
        dΣ = @thunk(-Δ.*(cos.(x .- μ) .- 1).*Σinv.^2)
        return NoTangent(), dμ, dΣ, dx
    end
    return s, _cp_logpdf_pullback
end

    # @simd for i in eachindex(μ, Σ)
    #     Σinv = inv(Σ[i])
    #     si,ci = sincos(x[i] - μ[i])
    #     s += (ci-1)*Σinv
    #     dμ[i] = si*Σinv
    #     dx[i] = -si*Σinv
    #     dΣ[i] = -(ci - 1)*Σinv^2
    # end
