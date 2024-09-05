export ClosurePhaseLikelihood

"""
    ClosurePhaseLikelihood(μ, Σ)

Construct the Gaussian approximate closure phase likelihood distribution.
The distribution used here is not the exact distribution for closure phases since this
becomes quite complicated (especially for correlated covariances).
For our approximation we take the Gaussian around the complex exponential of the phase, that
is the unormalized likelihood is given by

```math
    \\log\\mathcal{L} = -\\frac{1}{2}(e^{i\\theta} - e^{i\\mu})^T\\Sigma^{-1}(e^{i\\theta} - e^{i\\mu})
```

If Σ is diagonal then this reduces to a bunch of independent Von Mises distributions, and if
Σ is dense, this becomes a complicated multi-variate extension of the Von Mises
distribution[^1].

# Parameters
  - `μ`: The mean closure phase, which is usually computed from some VLBI model
  - `Σ`: The measurement covariance matrix, which is usually computed directly from the data.
         Note that `Σ` can either be a matrix or a vector. If `Σ` is a vector then we interpret
         `Σ` to represent the diagonal of the covariance matrix.

!!! warning
    Note that for the dense Σ, the likelihood is not normalized properly since the normalization
    is not analytically tractable. This is not a problem for sampling since the normalizing constant
    does not depend on `μ` so it does not *usually* impact parameter estimation. However, if you are
    including terms that modify the covariance `Σ` then this likelihood is wrong and could give
    biased results. If you want to fit noise terms in Σ please either use the diagonal approximation
    to the likelihood, or **better yet** fit complex visibilities directly.


[^1]: This distribution is defined on the n torus where n is the number of closure phases.
"""
struct ClosurePhaseLikelihood{V1,V2<:Union{AbstractVector, AbstractMatrix},W} <: AbstractVLBIDistributions
    """mean closure phase"""
    μ::V1
    """
    Covariance of the measurement (either diagonal or not)
    """
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
    Σpd = CholeskyFactor(Σ)
    return ClosurePhaseLikelihood(μ, Σpd)
end

function ClosurePhaseLikelihood(μ::AbstractVector, Σ::CholeskyFactor)
    lognorm = _gaussnorm(μ, Σ)
    return ClosurePhaseLikelihood(μ, Σ, lognorm)
end

function _closurephasenorm(μ, Σ::AbstractVector)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    return -n*log2π - sum(x->log(besseli0x(inv(x))), Σ)
end


# # We mark the norms as non-differentiable. Why? Because they are wrong anyways!
# ChainRulesCore.@non_differentiable _closurephasenorm(μ, Σ::CholeskyFactor)

function unnormed_logpdf(d::ClosurePhaseLikelihood{V,P}, x) where {V, P<:CholeskyFactor}
    return _cp_logpdf_full(d.μ, d.Σ, x)
end

function _cp_logpdf_full(μ, Σ, x)
    dθ = cis.(x) .- cis.(μ)
    z = _chi2(dθ, Σ)
    return -z
end


function Distributions._rand!(rng::Random.AbstractRNG, d::ClosurePhaseLikelihood{<:AbstractVector, <:CholeskyFactor}, x::AbstractVector)
    randn!(rng, x)
    chol = d.Σ
    r = _color!(zero(x), chol, x)
    x .= d.μ .+ r
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
