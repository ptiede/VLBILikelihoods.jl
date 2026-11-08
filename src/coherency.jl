export CoherencyLikelihood

"""
    CoherencyLikelihood(μ, w)

The likelihood of coherency matrices: an independent complex Gaussian for each of the four
feed products of each value, with mean `μ` and weight `w = 1/σ²` (see
[`ComplexVisLikelihood`](@ref)). `μ` and `w` are arrays of 2×2 static matrices (best as
`StructArray`s), or dense numeric arrays whose last two dims are the feeds. A product with weight `0` is unmeasured and adds nothing.
"""
struct CoherencyLikelihood{V1, V2, W} <: AbstractVLBIDistributions
    μ::V1
    w::V2
    lognorm::W
end

Base.length(d::CoherencyLikelihood) = length(d.μ)
Base.size(d::CoherencyLikelihood) = size(d.μ)
Base.eltype(d::CoherencyLikelihood) = eltype(d.μ)
Dists.insupport(d::CoherencyLikelihood, x) = true
Dists.mean(d::CoherencyLikelihood) = d.μ

function CoherencyLikelihood(μ::AbstractArray, w::AbstractArray)
    μs, ws = _components(μ), _components(w)
    foreach(_checkweights, μs, ws)
    return CoherencyLikelihood(μ, w, sum(_cvisnorm, ws))
end

# The four elements of a set of coherency matrices, in column-major order: the components of a
# `StructArray` of 2×2 matrices, or the slabs of dense storage whose last two dims are the feeds.
_components(x::StructArray{<:SA.StaticMatrix{2, 2}}) = values(StructArrays.components(x))
_components(x::AbstractArray{<:SA.StaticMatrix{2, 2}}) = _components(StructArray(x))
function _components(x::AbstractArray{<:Number, N}) where {N}
    N >= 2 && size(x)[(N - 1):N] == (2, 2) ||
        throw(DimensionMismatch("dense coherencies need trailing feed dims (2, 2), got size $(size(x))"))
    return ntuple(k -> selectdim(selectdim(x, N, cld(k, 2)), N - 1, mod1(k, 2)), Val(4))
end

function unnormed_logpdf(d::CoherencyLikelihood, x::AbstractArray)
    μs, ws, xs = _components(d.μ), _components(d.w), _components(x)
    return -sum(map(_weightedchi2, μs, ws, xs))
end

function Dists._rand!(rng::Random.AbstractRNG, d::CoherencyLikelihood, x::AbstractArray)
    for (xk, μk, wk) in zip(_components(x), _components(d.μ), _components(d.w))
        Dists._rand!(rng, ComplexVisLikelihood(μk, wk, zero(real(eltype(wk)))), xk)
    end
    return x
end
