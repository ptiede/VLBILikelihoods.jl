myzero(x) = zero(x)
myzero(::Type{<:SMatrix{2,2,T}}) where {T} = zero(MMatrix{2,2,T,4})
myzero(s::AbstractVector{<:SMatrix{2,2,T}}) where {T} = zeros(MMatrix{2,2,T,4}, length(s))

function _unnormed_logpdf_μΣ(μ, Σ, x)
    s = myzero(eltype(Σ))
    @simd for i in eachindex(μ, Σ)
        @. s += -abs2(x[i] - μ[i])*inv(Σ[i])
    end
    return s/2
end

function ChainRulesCore.rrule(::typeof(_unnormed_logpdf_μΣ), μ, Σ, x)
    s = myzero(eltype(Σ))
    dx = myzero(x)
    dμ = myzero(μ)
    dΣ = myzero(Σ)
    s = _unnormed_logpdf_μΣ(μ, Σ, x)
    @simd for i in eachindex(μ, Σ)
        Δx = abs.(x[i] - μ[i])
        Σinv = inv.(Σ[i])
        s .+= -Δx.^2 .* Σinv
        dμ[i] .= dx[i] .= -Δx.*Σinv
        dΣ[i] .= Δx.^2 .* Σinv.^2 ./2
    end
    function _unnormed_logpdf_μΣ_pullback(Δ)
        for i in eachindex(dμ, dx, dΣ)
            dμ[i] .*= -Δ
            dx[i] .*= Δ
            dΣ[i] .*= Δ
        end
        Σinv = map(x->inv.(x), Σ)
        Δx = abs.(x .- μ)
        dμ = Δ.*Δx.*Σinv
        return NoTangent(), dμ, dΣ, dx
    end

    return s./2, _unnormed_logpdf_μΣ_pullback
end


function _gaussnorm(μ, Σ)
    @assert length(μ) == length(Σ) "Mean and std. dev. vector are not the same length"
    n = length(μ)
    logw = -n/2*log2π - sum(log, Σ)/2
    return logw
end
