export ComplexVisLikelihood

"""
    ComplexVisLikelihood(μ::AbstractArray{<:Complex}, w::AbstractArray{<:Real})

The likelihood of complex visibilities: independent Gaussians with mean `μ` and weight
`w = 1/σ²`, where `σ` is the noise of the real and of the imaginary part. `μ` and `w` are
arrays of any shape with the same axes. A value with weight `0` is unmeasured and adds
nothing; its data must still be finite.

## Parameters
- `μ`: the mean visibilities, usually from a model.
- `w`: the weights, from the data.
"""
struct ComplexVisLikelihood{V1, V2, W} <: AbstractVLBIDistributions
    μ::V1
    w::V2
    lognorm::W
end

Base.length(d::ComplexVisLikelihood) = length(d.μ)
Base.size(d::ComplexVisLikelihood) = size(d.μ)
Base.eltype(d::ComplexVisLikelihood) = eltype(d.μ)
Dists.insupport(d::ComplexVisLikelihood, x) = true
Dists.mean(d::ComplexVisLikelihood) = d.μ
"""
    var(d::ComplexVisLikelihood)

`E|x - μ|² = 2/w`, the variance of each complex value; `Inf` where nothing was measured.
"""
Dists.var(d::ComplexVisLikelihood) = 2 .* inv.(d.w)

function ComplexVisLikelihood(μ::AbstractArray, w::AbstractArray)
    _checkweights(μ, w)
    return ComplexVisLikelihood(μ, w, _cvisnorm(w))
end

# Two real Gaussian components per value.
_cvisnorm(w) = 2 * _lognormweights(w)

unnormed_logpdf(d::ComplexVisLikelihood, x::AbstractArray) = -_weightedchi2(d.μ, d.w, x)

function Dists._rand!(rng::Random.AbstractRNG, d::ComplexVisLikelihood, x::AbstractArray)
    σ = _stddev(d.w)
    T = real(eltype(x))
    x .= d.μ .+ complex.(randn(rng, T, size(x)), randn(rng, T, size(x))) .* σ
    return x
end
