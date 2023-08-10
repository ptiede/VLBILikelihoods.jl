function _unnormed_logpdf_μΣ(μ, Σ, x)
    s = zero(eltype(Σ))
    z = zero(s)
    for i in eachindex(μ, Σ)
        # tmp = ifelse(!(isnan(x[i])&&isnan(Σ[i])), -abs2(x[i] - μ[i])*inv(Σ[i]), z)
        tmp = ifelse(!(isnan(x[i]) || isnan(Σ[i])), -abs2(x[i] - μ[i])*inv(Σ[i]), z)
        s += tmp
    end
    return s/2
end


function _gaussnorm(μ, Σ::AbstractVector)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    logw = -n/2*log2π - sum(log, filter!(!isnan, Σ))/2
    return logw
end

function _gaussnorm(μ, Σ::CholeskyFactor)
    @assert length(μ) == size(Σ,1) "Mean and Cov vector are not the same dimension"
    n = length(μ)
    ldet = logdet(Σ)
    return -n/2*log2π - ldet/2
end

# These will be removed when https://github.com/JuliaStats/Distributions.jl/pull/1554
# is finally merged
function ChainRulesCore.rrule(::typeof(_gaussnorm), μ, Σ::CholeskyFactor)
    y = _gaussnorm(μ,  Σ)
    function _gaussnorm_pullback(Δ)
        # invΣ = inv(Σ.cov)
        # ∂Σ = (unthunk(Δ) / (-2)) * invΣ
        return NoTangent(), NoTangent(), NoTangent()
    end
    return y, _gaussnorm_pullback
end

_chi2(dx, Σ) = abs(dot(dx, Σ\dx))/2
