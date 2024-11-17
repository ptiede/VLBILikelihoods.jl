@noinline function _unnormed_logpdf_μΣ(μ, Σ, x)
    z = zero(eltype(Σ))
    s = sum(zip(μ, Σ, x); init=zero(eltype(Σ))) do (μs, Σs, xs)
            if isnan(xs) || isnan(Σs)
                return z
            end
            return -abs2(xs - μs)/Σs
        end

    # s = zero(eltype(Σ))
    # z = zero(s)
    # for i in eachindex(x, μ, Σ)
    #     if isnan(μ[i])
    #         return NaN #short circuit because this should never happen
    #     end

    #     if isnan(x[i]) || isnan(Σ[i])
    #         continue
    #     else 
    #         s += -abs2(x[i] - μ[i])*inv(Σ[i])
    #     end
    #     # tmp = ifelse(!(isnan(x[i]) || isnan(Σ[i])), -abs2(x[i] - μ[i])*inv(Σ[i]), z)
    #     # s += tmp
    # end
    return s/2
end


# function _gaussnorm(μ, Σ::AbstractVector)
#     @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
#     n = length(μ)
#     logw = -n/2*log2π - sum(log, filter!(!isnan, Σ))/2
#     return logw
# end

function _gaussnorm(μ, Σ::AbstractVector)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    logw = -n*convert(eltype(Σ), log2π)/2
    logs = sum(Σ) do s
            return ifelse(!isnan(s), log(s), (zero(eltype(Σ))))
    end
    return logw - logs/2
end


function _gaussnorm(μ, Σ::CholeskyFactor)
    @assert length(μ) == size(Σ,1) "Mean and Cov vector are not the same dimension"
    n = length(μ)
    ldet = logdet(Σ)
    return -n*convert(typeof(ldet), log2π)/2 - ldet/2
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
