function _unnormed_logpdf_μΣ(μ, Σ, x)
    s = zero(eltype(Σ))
    @simd for i in eachindex(μ, Σ)
        s += -abs2(x[i] - μ[i])*inv(Σ[i])
    end
    return s/2
end


function _gaussnorm(μ, Σ::AbstractVector)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    logw = -n/2*log2π - sum(log, Σ)/2
    return logw
end

function _gaussnorm(μ, Σ::PDMat)
    @assert length(μ) == size(Σ,1) "Mean and Cov vector are not the same dimension"
    n = length(μ)
    ldet = logdet(Σ)
    return -n/2*log2π - ldet/2
end

function ChainRulesCore.rrule(::typeof(_gaussnorm), μ, Σ::PDMat)
    y = _gaussnorm(μ,  Σ)
    function _gaussnorm_pullback(Δ)
        invΣ = inv(Σ)
        ∂Σ = (unthunk(Δ) / (-2)) * invΣ.mat
        return NoTangent(), NoTangent(), ∂Σ
    end
    return y, _gaussnorm_pullback
end



_chi2(dx, Σ) = invquad(Σ, dx)/2
