function _unnormed_logpdf_μΣ(μ, Σ, x)
    s = zero(eltype(Σ))
    @simd for i in eachindex(μ, Σ)
        s += -abs2(x[i] - μ[i])*inv(Σ[i])
    end
    return s/2
end

function ChainRulesCore.rrule(::typeof(_unnormed_logpdf_μΣ), μ, Σ, x)
    s = zero(eltype(Σ))
    dx = zero(x)
    dμ = zero(μ)
    dΣ = zero(Σ)
    @simd for i in eachindex(μ, Σ)
        Δx = abs(x[i] - μ[i])
        Σinv = inv(Σ)
        s += -Δx^2*Σinv
        dμ[i] = dx[i] = -Δx*Σinv
        dΣ[i] = Δx^2*Σinv^2/2
    end

    function _unnormed_logpdf_μΣ_pullback(Δ)
        dμ .= -Δ.*dμ
        dx .= Δ.*dx
        dΣ .= Δ.*dΣ
        return NoTangent(), dμ, dΣ, dx
    end

    return s/2, _unnormed_logpdf_μΣ_pullback
end


function _gaussnorm(μ, Σ)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    logw = -n/2*log2π - sum(log, Σ)/2
    return logw
end
