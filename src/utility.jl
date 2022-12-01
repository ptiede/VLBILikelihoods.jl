function _unnormed_logpdf_μΣ(μ, Σ, x)
    s = zero(eltype(Σ))
    @simd for i in eachindex(μ, Σ)
        s += -abs2(x[i] - μ[i])*inv(Σ[i])
    end
    return s/2
end

function ChainRulesCore.rrule(::typeof(_unnormed_logpdf_μΣ), μ, Σ, x)
    s = _unnormed_logpdf_μΣ(μ, Σ, x)

    function _unormed_logpdf_μΣ_pullback(Δ)
        Δx = x .- μ
        invΣ = inv.(Σ)
        dμ = @thunk(Δ.*Δx.*invΣ)
        dx = @thunk(-Δ.*Δx.*invΣ)
        dΣ = @thunk(Δ.*abs2.(Δx).*invΣ.^2/2)
        return NoTangent(), dμ, dΣ, dx
    end

    return s, _unormed_logpdf_μΣ_pullback
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
    return -n/2*log2π - ldet
end

function ChainRulesCore.rrule(::typeof(_gaussnorm), μ, Σ::PDMat)
    y = _gaussnorm(μ,  Σ)
    function _gaussnorm_pullback(Δ)
        ∂Σ = (unthunk(Δ) / (-2)) * inv(Σ)
        return NoTangent(), NoTangent(), ∂Σ
    end
    return y, _gaussnorm_pullback
end


_chi2(dx, Σ) = invquad(Σ, dx)

# Temp until https://github.com/JuliaStats/Distributions.jl/pull/1554 is merged
function ChainRulesCore.rrule(::typeof(_chi2), dx::AbstractVector, Σ)
    y = _chi2(dx, Σ)
    z = Σ \ dx
    function _chi2_pullback(Δ)
        ∂ = ChainRulesCore.unthunk(Δ)
        ∂x = 2 * ∂ * z
        ∂Σ = ChainRulesCore.@thunk(begin
            ∂J = ∂ * dx * dx'
            - (Σ \ ∂J) / Σ
        end)
        return (ChainRulesCore.NoTangent(), ∂x, ∂Σ)
    end
    return y, _chi2_pullback
end
