function ChainRulesCore.rrule(::typeof(_unnormed_logpdf_μΣ), μ::AbstractVector, Σ::AbstractVector, x::AbstractVector)
    s = _unnormed_logpdf_μΣ(μ, Σ, x)
    pμ = ProjectTo(μ)
    pΣ = ProjectTo(Σ)
    px = ProjectTo(x)
    function _unormed_logpdf_μΣ_pullback(Δ)
        Δx = x .- μ
        Δx .= ifelse.(Base.:!.(isnan.(Δx)), Δx, zero(eltype(Δx)))
        invΣ = inv.(Σ)
        invΣ .=  ifelse.(Base.:!.(isnan.(invΣ)), invΣ, zero(eltype(invΣ)))
        dμ = @thunk(pμ(Δ.*Δx.*invΣ))
        dx = @thunk(px(-Δ.*Δx.*invΣ))
        dΣ = @thunk(pΣ(Δ.*abs2.(Δx).*invΣ.^2/2))
        return NoTangent(), dμ, dΣ, dx
    end

    return s, _unormed_logpdf_μΣ_pullback
end

# Temp until https://github.com/JuliaStats/Distributions.jl/pull/1554 is merged
function ChainRulesCore.rrule(::typeof(_chi2), dx::AbstractVector, Σ)
    y = _chi2(dx, Σ)
    z = Σ \ dx
    function _chi2_pullback(Δ)
        ∂ = ChainRulesCore.unthunk(Δ)
        ∂x = ∂ * z
        ∂Σ = ZeroTangent()#ChainRulesCore.@thunk(begin
            #∂J = ∂ * dx * dx'./2
            #- ((Σ \ ∂J) / Σ)./2
        #end)
        return (ChainRulesCore.NoTangent(), ∂x, ∂Σ)
    end
    return y, _chi2_pullback
end


function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, T::Type{<:Union{AmplitudeLikelihood, ClosurePhaseLikelihood}}, μ::AbstractVector, Σ::AbstractMatrix)
    Σpd = CholeskyFactor(Σ)
    d = T(μ, Σpd)

    # get the normalization from the rrule
    function _AmplitudeNormal_pullback(Δ)
        Δlg = last(rrule_via_ad(config, _gaussnorm, μ, Σpd)[2](Δ.lognorm))
        Δμ = Δ.μ
        ΔΣ = Δ.Σ + Δlg
        return NoTangent(), Δμ, ΔΣ
    end
    return d, _AmplitudeNormal_pullback
end

function ChainRulesCore.rrule(::Type{<:AmplitudeLikelihood}, μ::AbstractVector, Σ::AbstractVector)
    lognorm = AmplitudeLikelihood(μ, Σ, _gaussnorm(μ, Σ))
    function _AmplitudeLikelihood_pullback(Δ)
        d = unthunk(Δ)
        Δμ = @thunk(d.μ)
        ΔΣ = @thunk(d.Σ .- d.lognorm.*inv.(Σ)/2)
        return NoTangent(), Δμ, ΔΣ
    end
    return lognorm, _AmplitudeLikelihood_pullback
end

function ChainRulesCore.rrule(::typeof(_closurephasenorm), μ, Σ::AbstractVector)
    v =zero(eltype(Σ))
    n = length(Σ)
    dΣ = zero(Σ)
    for i in eachindex(Σ)
        κ = inv(Σ[i])
        i0 = besseli0x(κ)
        i1 = besseli1x(κ)
        v += log(i0)
        dΣ[i] = (i1/i0-1)*κ^2
    end
    function _closurephasenorm_pullback(Δ)
       ΔΣ = Δ.*dΣ
        return NoTangent(), ZeroTangent(), ΔΣ
    end
    return -n*log2π - v, _closurephasenorm_pullback
end

function ChainRulesCore.rrule(::typeof(_cp_logpdf), μ::AbstractVector, Σ::AbstractVector, x::AbstractVector)
    s = _cp_logpdf(μ, Σ, x)
    pμ = ProjectTo(μ)
    pΣ = ProjectTo(Σ)
    px = ProjectTo(x)

    function _cp_logpdf_pullback(Δ)
        Σinv = inv.(Σ)
        ss = sin.(x .- μ)
        dμ = @thunk(pμ(Δ.*ss.*Σinv))
        dx = @thunk(px(-Δ.*ss.*Σinv))
        dΣ = @thunk(pΣ(-Δ.*(cos.(x .- μ) .- 1).*Σinv.^2))
        return NoTangent(), dμ, dΣ, dx
    end
    return s, _cp_logpdf_pullback
end


function ChainRulesCore.rrule(::Type{<:ComplexVisLikelihood}, μ::AbstractVector, Σ::AbstractVector)
    lognorm = ComplexVisLikelihood(μ, Σ)
    function _ComplexVisLikelihood_pullback(Δu)
        Δ = unthunk(Δu)
        Δμ = @thunk(Δ.μ)
        ΔΣ = @thunk(Δ.Σ .- Δ.lognorm.*inv.(Σ))
        return NoTangent(), Δμ, ΔΣ
    end
    return lognorm, _ComplexVisLikelihood_pullback
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(unnormed_logpdf),
    d::CoherencyLikelihood{<:StructVector{<:SA.StaticMatrix{2,2,T}}, <:StructVector{<:SA.StaticMatrix{2,2,S}}},
    x::StructVector{<:SA.StaticMatrix{2,2,X}}
    ) where {T, S, X}
    μs = StructArrays.components(d.μ)
    Σs = StructArrays.components(d.Σ)
    xs = StructArrays.components(x)

    l1, dl1 = rrule_via_ad(config, _unnormed_logpdf_μΣ, μs[1], Σs[1], xs[1])
    l2, dl2 = rrule_via_ad(config, _unnormed_logpdf_μΣ, μs[2], Σs[2], xs[2])
    l3, dl3 = rrule_via_ad(config, _unnormed_logpdf_μΣ, μs[3], Σs[3], xs[3])
    l4, dl4 = rrule_via_ad(config, _unnormed_logpdf_μΣ, μs[4], Σs[4], xs[4])

    ll = l1 + l2 + l3 + l4

    function _unnormed_lpdf_coherency(Δ)
        Δl1 = dl1(Δ)
        Δl2 = dl2(Δ)
        Δl3 = dl3(Δ)
        Δl4 = dl4(Δ)
        dμ = StructVector{SMatrix{2,2,T,4}}((unthunk(Δl1[2]), unthunk(Δl2[2]), unthunk(Δl3[2]), unthunk(Δl4[2])))
        dΣ = StructVector{SMatrix{2,2,S,4}}((unthunk(Δl1[3]), unthunk(Δl2[3]), unthunk(Δl3[3]), unthunk(Δl4[3])))
        dx = StructVector{SMatrix{2,2,X,4}}((unthunk(Δl1[4]), unthunk(Δl2[4]), unthunk(Δl3[4]), unthunk(Δl4[4])))
        return NoTangent(), Tangent{typeof(d)}(μ=dμ, Σ=dΣ, lognorm=ZeroTangent()), dx
    end

    return ll, _unnormed_lpdf_coherency
end

function _comp_coherency_pullback(Δ, Σxx)
    return - Δ.*inv.(Σxx)
end

function ChainRulesCore.rrule(::typeof(_coherencynorm), μ, Σ::StructVector)
    s = _coherencynorm(μ, Σ)
    function _coherencynorm_pullback(Δ)
        Σc = StructArrays.components(Σ)
        ΔΣ = _comp_coherency_pullback.(Ref(Δ), Σc)

        return NoTangent(), ZeroTangent(), StructVector{eltype(Σ)}(ΔΣ)
    end
    return s, _coherencynorm_pullback
end

EnzymeRules.inactive_type(::Type{<:CholeskyFactor}) = true


## We add this untile Enzyme fixes this issue upstream
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(\)}, ::Type{RT}, 
                                      c::Annotation{<:CholeskyFactor}, v::Annotation{<:AbstractArray}) where {RT}

    if !(typeof(c) <: Const)
        throw(ArgumentError("CholeskyFactor must be a Const"))
    end

    cache_c = if EnzymeRules.overwritten(config)[2]
        c.val
    else
        c.val
    end

    res = c.val \ v.val

    cache_v = if EnzymeRules.overwritten(config)[3]
        copy(v.val)
    else
        v.val
    end

    cache_res = if EnzymeRules.needs_primal(config)
        copy(res)
    else
        res
    end


    primal = if EnzymeRules.needs_primal(config)
        res
    else
        nothing
    end

    dres = if EnzymeRules.width(config) == 1
        zero(res)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zero(res)
        end
    end

    retres = if EnzymeRules.needs_primal(config)
        res
    else
        nothing
    end

    cache = NamedTuple{(Symbol("1"),Symbol("2"), Symbol("3"), Symbol("4")), Tuple{
        eltype(RT),
        EnzymeRules.needs_shadow(config) ? (EnzymeRules.width(config) == 1 ? eltype(RT) : NTuple{EnzymeRules.width(config), eltype(RT)}) : Nothing,
        typeof(cache_c),
        typeof(cache_v)
        }}(
        (cache_res, dres, cache_c, cache_v)
    )

    # For EnzymeCore 0.8
    return EnzymeRules.AugmentedReturn{
        EnzymeRules.primal_type(config, RT),
        EnzymeRules.shadow_type(config, RT),
        typeof(cache)
    }(retres, dres, cache)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(\)}, ::Type{RT}, cache, 
                             c::Annotation{<:CholeskyFactor}, v::Annotation{<:AbstractArray}) where {RT}

    res, dres, cache_c, cache_v = cache
    
    if !(typeof(c) <: Const)
        throw(ArgumentError("CholeskyFactor must be a Const"))
    end

    
    if !EnzymeRules.overwritten(config)[3]
        cache_v = v.val
    end

    if !EnzymeRules.overwritten(config)[2]
        cache_c = c.val
    end

    if EnzymeRules.width(config) == 1
        dress = (dres,)
    else
        dress = dres
    end

    dvs = if EnzymeRules.width(config) == 1
            (v.dval,)
    else
        v.dval
    end

    for (dv, dres) in zip(dvs, dress)
        z = adjoint(c.val.cho) \ dres
        dv .+= z
        dres .= zero(eltype(dres))
    end

    return (nothing, nothing)
end
