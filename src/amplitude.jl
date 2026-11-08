export AmplitudeLikelihood, RiceAmplitudeLikelihood

"""
    AmplitudeLikelihood(μ, w::AbstractArray)
    AmplitudeLikelihood(μ, Σ::CholeskyFactor)

The Gaussian likelihood of amplitudes with mean `μ`: independent, with weights `w = 1/σ²` of the
axes of `μ` (a value with weight `0` adds nothing), or correlated, with the covariance
`Σ` given as a [`CholeskyFactor`](@ref).

!!! warning
    The amplitudes are not debiased. This likelihood is significantly biased from the Rice
    distribution for SNR `μ/σ < 2`; use [`RiceAmplitudeLikelihood`](@ref) there.
"""
struct AmplitudeLikelihood{V1, V2, W} <: AbstractVLBIDistributions
    μ::V1
    w::V2
    lognorm::W
end

Base.length(d::AmplitudeLikelihood) = length(d.μ)
Base.size(d::AmplitudeLikelihood) = size(d.μ)
Base.eltype(d::AmplitudeLikelihood) = promote_type(eltype(d.μ), eltype(d.w))
Dists.insupport(::AmplitudeLikelihood, x) = true
Dists.mean(d::AmplitudeLikelihood) = d.μ

function AmplitudeLikelihood(μ::AbstractArray, w::AbstractArray)
    _checkweights(μ, w)
    return AmplitudeLikelihood(μ, w, _lognormweights(w))
end

AmplitudeLikelihood(μ::AbstractVector, Σ::CholeskyFactor) = AmplitudeLikelihood(μ, Σ, _gaussnorm(μ, Σ))

unnormed_logpdf(d::AmplitudeLikelihood, x::AbstractArray) = -_weightedchi2(d.μ, d.w, x)
unnormed_logpdf(d::AmplitudeLikelihood{V, <:CholeskyFactor}, x::AbstractVector) where {V} = _amp_logpdf_full(d.μ, d.w, x)

function _amp_logpdf_full(μ, Σ, x)
    _checkdata(μ, x)
    return -_chi2(x .- μ, Σ)
end

function Distributions._rand!(rng::Random.AbstractRNG, d::AmplitudeLikelihood, x::AbstractArray{<:Real})
    x .= d.μ .+ randn(rng, eltype(x), size(x)) .* _stddev(d.w)
    return x
end

function Distributions._rand!(rng::Random.AbstractRNG, d::AmplitudeLikelihood{<:AbstractVector, <:CholeskyFactor}, x::AbstractVector{<:Real})
    randn!(rng, x)
    r = zero(x)
    _color!(r, d.w, x)
    x .= d.μ .+ r
    return x
end

"""
    RiceAmplitudeLikelihood(μ, w::AbstractArray)

The Rice likelihood of amplitudes with model amplitudes `μ` and weights `w = 1/σ²` of the axes
of `μ`, `σ` the noise of each component of the complex visibility. A value with weight `0` is
unmeasured and adds nothing; its data must be finite and non-negative.

!!! note
    This is the correct distribution of visibility amplitudes, but it is slower than the
    Gaussian approximation [`AmplitudeLikelihood`](@ref). Low-SNR data need it to be unbiased.
"""
struct RiceAmplitudeLikelihood{V1, V2, W} <: AbstractVLBIDistributions
    μ::V1
    w::V2
    lognorm::W
end

Base.length(d::RiceAmplitudeLikelihood) = length(d.μ)
Base.size(d::RiceAmplitudeLikelihood) = size(d.μ)
Base.eltype(d::RiceAmplitudeLikelihood) = promote_type(eltype(d.μ), eltype(d.w))
Dists.insupport(d::RiceAmplitudeLikelihood, x::AbstractVector) = _insupport(d, x)
_insupport(::RiceAmplitudeLikelihood, x) = all(>=(0), x)

function RiceAmplitudeLikelihood(μ::AbstractArray, w::AbstractArray)
    _checkweights(μ, w)
    return RiceAmplitudeLikelihood(μ, w, _ricenorm(w))
end

# `Σ_{w > 0} log w`
function _ricenorm(w)
    T = float(real(eltype(w)))
    return sum(w; init = zero(T)) do wi
        measured = wi > 0
        return ifelse(measured, log(ifelse(measured, wi, one(wi))), zero(T))
    end
end

_L12(x) = exp(x / 2) * ((1 - x) * besseli0(-x / 2) - x * besseli1(-x / 2))

function unnormed_logpdf(d::RiceAmplitudeLikelihood, x::AbstractArray)
    _checkdata(d.μ, x)
    return sum(zip(d.μ, d.w, x); init = zero(float(real(eltype(d.w))))) do (μ, w, xx)
        measured = w > 0
        # an unmeasured value is evaluated at x = 1 and dropped, so no log(0) reaches the gradient
        l = lpdf_rice(ifelse(measured, xx, one(xx)), μ, w)
        return ifelse(measured, l, zero(l))
    end
end

# besseli0x keeps high-SNR points finite.
lpdf_rice(x, μ, w) = -(x^2 + μ^2) * w / 2 + log(x * besseli0x(x * μ * w)) + x * μ * w

Dists.mean(d::RiceAmplitudeLikelihood) = sqrt.(inv.(d.w) .* π / 2) .* _L12.(-d.μ .^ 2 .* d.w ./ 2)
function Dists.var(d::RiceAmplitudeLikelihood)
    T = eltype(d)
    Σ = inv.(d.w)
    return 2 * Σ .+ d.μ .^ 2 - (T(π) .* Σ / 2) .* _L12.(-d.μ .^ 2 .* d.w ./ 2) .^ 2
end
Dists.cov(d::RiceAmplitudeLikelihood) = Diagonal(vec(Dists.var(d)))

function Distributions._rand!(rng::Random.AbstractRNG, d::RiceAmplitudeLikelihood, x::AbstractArray{<:Real})
    σ = _stddev(d.w)
    T = eltype(x)
    x .= abs.(complex.(randn(rng, T, size(x)) .* σ .+ d.μ, randn(rng, T, size(x)) .* σ))
    return x
end
