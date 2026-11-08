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
