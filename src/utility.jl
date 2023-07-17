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

function _gaussnorm(μ, Σ::AbstractPDMat)
    @assert length(μ) == size(Σ,1) "Mean and Cov vector are not the same dimension"
    n = length(μ)
    ldet = logdet(Σ)
    return -n/2*log2π - ldet/2
end

# These will be removed when https://github.com/JuliaStats/Distributions.jl/pull/1554
# is finally merged
function ChainRulesCore.rrule(::typeof(_gaussnorm), μ, Σ::AbstractPDMat)
    y = _gaussnorm(μ,  Σ)
    function _gaussnorm_pullback(Δ)
        invΣ = inv(Σ)
        ∂Σ = (unthunk(Δ) / (-2)) * invΣ.mat
        return NoTangent(), NoTangent(), ∂Σ
    end
    return y, _gaussnorm_pullback
end

@noinline chol(Σ::PDSparseMat) = Σ.chol

_chi2(dx, Σ) = abs(dot(dx, chol(Σ)\dx))/2
