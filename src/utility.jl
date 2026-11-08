function _checkweights(μ, w)
    axes(μ) == axes(w) ||
        throw(DimensionMismatch("the weights need the axes of the mean $(axes(μ)), got $(axes(w)); give a dense covariance as `CholeskyFactor(Σ)`"))
    all(x -> isfinite(x) && x >= 0, w) || throw(ArgumentError("weights must be finite and non-negative"))
    return nothing
end

function _checkdata(μ, x)
    axes(x) == axes(μ) || throw(DimensionMismatch("the data have axes $(axes(x)); the likelihood has $(axes(μ))"))
    return nothing
end

# The real type of a sum over terms combining `μ`, `w` and `x`, so the accumulator keeps one type.
_sumtype(μ, w, x) = float(real(promote_type(eltype(μ), eltype(w), eltype(x))))

# `½ Σ w |x - μ|²`. A value of weight 0 contributes nothing, so its data must be finite.
function _weightedchi2(μ, w, x)
    _checkdata(μ, x)
    s = sum(zip(μ, w, x); init = zero(_sumtype(μ, w, x))) do (m, wi, xi)
        return wi * abs2(xi - m)
    end
    return s / 2
end

# `Σ_{w > 0} (log w - log 2π) / 2`: the normalization of one real Gaussian component per value.
function _lognormweights(w)
    T = float(real(eltype(w)))
    return sum(w; init = zero(T)) do wi
        measured = wi > 0
        return ifelse(measured, (log(ifelse(measured, wi, one(wi))) - T(log2π)) / 2, zero(T))
    end
end

# The noise standard deviation of each value, 0 where nothing was measured.
_stddev(w) = @. ifelse(w > 0, inv(sqrt(ifelse(w > 0, w, one(w)))), zero(w))

function _gaussnorm(μ, Σ::CholeskyFactor)
    length(μ) == size(Σ, 1) ||
        throw(DimensionMismatch("the mean has length $(length(μ)) but the covariance is $(size(Σ))"))
    n = length(μ)
    ldet = logdet(Σ)
    return -n * convert(typeof(ldet), log2π) / 2 - ldet / 2
end

_chi2(dx, Σ) = abs(dot(dx, Σ \ dx)) / 2
