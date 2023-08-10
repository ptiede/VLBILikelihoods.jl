struct CholeskyFactor{T, P<:AbstractMatrix{T},C} <: AbstractMatrix{T}
    cov::P
    cho::C
end
CholeskyFactor(cov::AbstractMatrix) = CholeskyFactor(cov, cholesky(cov))
Base.parent(m::CholeskyFactor) = m.cov
Base.size(m::CholeskyFactor) = size(parent(m))
Base.getindex(m::CholeskyFactor, i::Int) = getindex(parent(m), i)
Base.getindex(m::CholeskyFactor, I::Vararg{Int, 2}) = getindex(parent(m), I...)
Base.IndexStyle(::Type{<:CholeskyFactor{T,M}}) where {T,M} = IndexStyle(M)
LinearAlgebra.cholesky(m::CholeskyFactor) = m

Base.:\(c::CholeskyFactor, v::AbstractVector) = c.cho\v

function _color!(r, Σ::CholeskyFactor, x)
    chol = Σ.cho
    mul!(r, chol.L, x)
    return r
end

function _color!(r, chol::CholeskyFactor{T, <:AbstractMatrix{T}, C}, x) where {T, C<:SuiteSparse.CHOLMOD.Factor}
    cho = chol.cho
    PtL = sparse(cho.L)[cho.p, :]
    mul!(r, PtL, x)
    return r
end
