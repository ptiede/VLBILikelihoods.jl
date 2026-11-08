export ClosurePhaseLikelihood

"""
    ClosurePhaseLikelihood(μ, w::AbstractArray)
    ClosurePhaseLikelihood(μ, Σ::CholeskyFactor)

The Gaussian approximation to the closure phase likelihood, taken around the complex
exponential of the phase:

```math
    \\log\\mathcal{L} = -\\frac{1}{2}(e^{i\\theta} - e^{i\\mu})^T\\Sigma^{-1}(e^{i\\theta} - e^{i\\mu})
```

With weights `w = 1/σ²` of the axes of `μ` the closure phases are independent von Mises
distributions (a value with weight `0` adds nothing). With a covariance `Σ`, given as a
[`CholeskyFactor`](@ref), the distribution is a correlated extension of the von Mises
distribution on the n-torus.

# Parameters
  - `μ`: the mean closure phases, usually from a model.
  - `w` or `Σ`: the measurement weights or covariance, from the data.

!!! warning
    With a covariance `Σ` the likelihood is not normalized: its normalization has no closed
    form. That does not matter when `Σ` is fixed, but it biases fits of noise terms in `Σ`; for
    those, use weights or fit complex visibilities.
"""
struct ClosurePhaseLikelihood{V1, V2, W} <: AbstractVLBIDistributions
    μ::V1
    w::V2
    lognorm::W
end

Base.length(d::ClosurePhaseLikelihood) = length(d.μ)
Base.size(d::ClosurePhaseLikelihood) = size(d.μ)
Base.eltype(d::ClosurePhaseLikelihood) = promote_type(eltype(d.μ), eltype(d.w))
Dists.insupport(d::ClosurePhaseLikelihood, x) = true

function ClosurePhaseLikelihood(μ::AbstractArray, w::AbstractArray)
    _checkweights(μ, w)
    return ClosurePhaseLikelihood(μ, w, _closurephasenorm(w))
end

ClosurePhaseLikelihood(μ::AbstractVector, Σ::CholeskyFactor) = ClosurePhaseLikelihood(μ, Σ, _gaussnorm(μ, Σ))

# `Σ_{w > 0} -(log 2π + log I0x(w))`, the von Mises normalization with concentration `w`.
function _closurephasenorm(w)
    T = float(real(eltype(w)))
    return sum(w; init = zero(T)) do wi
        measured = wi > 0
        return ifelse(measured, -T(log2π) - log(besseli0x(wi)), zero(T))
    end
end

unnormed_logpdf(d::ClosurePhaseLikelihood{V, <:CholeskyFactor}, x::AbstractVector) where {V} = _cp_logpdf_full(d.μ, d.w, x)

function _cp_logpdf_full(μ, Σ, x)
    _checkdata(μ, x)
    return -_chi2(cis.(x) .- cis.(μ), Σ)
end

function unnormed_logpdf(d::ClosurePhaseLikelihood, x::AbstractArray)
    _checkdata(d.μ, x)
    return sum(zip(d.μ, d.w, x); init = zero(_sumtype(d.μ, d.w, x))) do (μ, w, xx)
        return (cos(xx - μ) - 1) * w
    end
end

function Distributions._rand!(rng::Random.AbstractRNG, d::ClosurePhaseLikelihood{<:AbstractVector, <:CholeskyFactor}, x::AbstractVector{<:Real})
    randn!(rng, x)
    r = _color!(zero(x), d.w, x)
    x .= d.μ .+ r
    return x
end

function Distributions._rand!(rng::Random.AbstractRNG, d::ClosurePhaseLikelihood, x::AbstractArray{<:Real})
    x .= d.μ .+ randn(rng, eltype(x), size(x)) .* _stddev(d.w)
    return x
end
