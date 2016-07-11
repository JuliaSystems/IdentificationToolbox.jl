abstract AbstractIdMethod

abstract IterativeIdMethod <: AbstractIdMethod

abstract OneStepIdMethod <: AbstractIdMethod

function fval{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    method::IterativeIdMethod, data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2},
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  warn("Every IterativeIdMethod need to imlement the function fval")
  throw(DomainError())
end

function gradhessian!{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    method::IterativeIdMethod, data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2},
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  warn("Every IterativeIdMethod need to imlement the function gradhessian")
  throw(DomainError())
end
