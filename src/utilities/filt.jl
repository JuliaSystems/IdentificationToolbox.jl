

import Base: filt!, filt
#import DSP: filt

function _zerosi{M<:AbstractArray}(b::Vector{M}, a::Vector{M},T)
  m  = max(length(a), length(b)) - 1
  si = zeros(promote_type(eltype(eltype(b)), eltype(eltype(a)),T),
    size(a[1],1), m)
end

"""
    filt!(out, b, a, x, [si])
Same as [`filt`](:func:`filt`) but writes the result into the `out` argument, which may
alias the input `x` to modify it in-place.
"""
function filt!{M<:AbstractArray,T,S,N}(
  out::AbstractArray, b::Vector{M}, a::Vector{M},
  x::AbstractArray{T}, si::AbstractArray{S,N}=_zerosi(b,a,T))
  isempty(b) && throw(ArgumentError("filter vector b must be non-empty"))
  isempty(a) && throw(ArgumentError("filter vector a must be non-empty"))
  ~isdiag(a[1]) && throw(ArgumentError("filter vector vector a[1] must be diagonal"))
  any(a[1][i,i] == 0 for i = size(a[1],1))  &&
    throw(ArgumentError("filter vector a[1] must have nonzero diagonal elements"))
  if size(x,2) != size(out,2)
      throw(ArgumentError("output size $(size(out)) must match input size $(size(x))"))
  end

  as = length(a)
  bs = length(b)
  sz = max(as, bs)
  silen = sz - 1

  if size(si, 2) != silen
      throw(ArgumentError("initial state vector si must have max(length(a),length(b))-1 columns"))
  end

  # Filter coefficient normalization
  for i = 1:size(a[1],1)
    if a[1][i,i] != 1
      norml = a[1][i,i]
      for k = 1:length(a), j = 1:size(a[1],2)
        a[k][i,j] ./= norml
      end
      for k = 1:length(b), j = 1:size(b[1],2)
        b[k][i,j] ./= norml
      end
    end
  end

  # Pad the coefficients with zeros if needed
  if bs < sz
    bn = Array(eltype(b), sz)
    for i in eachindex(b)
      bn[i] = b[i]
    end
    for i = bs+1:sz
      bn[i] = zeros(similar(b[1]))
    end
    b = bn
  end
  if 1 < as < sz
    an = Array(eltype(a), sz)
    for i in eachindex(a)
      an[i] = a[i]
    end
    for i = as+1:sz
      an[i] = zeros(similar(a[1]))
    end
    a = an
  end

  if as > 1
    _filt_iir!(out, b, a, x, si)
  else
    _filt_fir!(out, b, x, si)
  end
end

function _zerosi{S,G}(b::PolyMatrix{S}, a::PolyMatrix{G}, T)
  m  = max(order(a), order(b))
  si = zeros(promote_type(S, G, T), m, size(a,1))
end

function filt{T,S,G}(b::PolyMatrix{T}, a::PolyMatrix{S},
  x::AbstractArray{G}, si=_zerosi(b, a, G))
  filt!(Array(promote_type(T, G, S), size(x,1),size(a,2)), b, a, x, si)
end

function filt!{H,T,S,G}(out::AbstractArray{H}, b::PolyMatrix{T},
  a::PolyMatrix{S}, x::AbstractArray{G}, si=_zerosi(b, a, G))

  #TODO
  # get filters to the same length

  _filt_iir!(out, b, a, x, si)
  return out
end

function _filt_iir!{T}(
  out::AbstractMatrix{T}, b::PolyMatrix{T}, a::PolyMatrix{T},
  x, si)
  silen = size(si,1)
  bc = coeffs(b)
  ac = coeffs(a)
  val = zeros(T,1,size(a,2))
  @inbounds @simd for i=1:size(x, 1)
    xi = view(x,i,:)
    val = si[1,:] + bc[0]*xi
    for j=1:(silen-1)
       si[j,:] = si[j+1,:,] + bc[j]*xi - ac[j]*val
    end
    si[silen,:] = bc[silen]*xi - ac[silen]*val
    out[i,:] = val
  end
end

function _filt_fir!{T,S}(
  out::AbstractMatrix{T}, b::PolyMatrix{T},
  x::AbstractArray{T}, si::AbstractMatrix{S})
  silen = size(si,1)
  bc = coeffs(b)
  @inbounds @simd for i=1:size(x, 1)
    xi = view(x,i,:)
    val = si[1,:] + bc[0]*xi
    for j=1:(silen-1)
      si[j,:] = si[j+1,:] + bc[j]*xi
    end
    si[silen,:] = bc[silen]*xi
    out[i,:] = val
  end
end

# polynomial filtering
function filt{T,S,G}(b::Poly{T}, a::Poly{S},
  x::AbstractVector{G}, si=zeros(promote_type(S, G, T), length(coeffs(a))-1))
  return filt(reverse(coeffs(b)), reverse(coeffs(a)), x, si)
end

function filt{T,S,G}(b::T, a::Poly{S},
  x::AbstractVector{G}, si=zeros(promote_type(S, G, T), length(coeffs(a))-1))
  out = Array(promote_type(T, G, S), size(x,1))
  _filt_ar!(out, a/b, x, si)
  return out
end

function _filt_ar!{T,S}(
  out::AbstractVector{T}, a::Poly{T},
  x::AbstractArray{T}, si::AbstractVector{S})
  silen = size(si,1)
  println(si)
  ac = reverse(coeffs(a))
  println(ac)
  val = zero(T)
  @inbounds @simd for i=1:size(x, 1)
    xi = x[i]
    val = si[1] + xi
    for j=1:(silen-1)
       si[j,:] = si[j+1,:,] - ac[j+1]*val
    end
    si[silen] = - ac[silen+1]*val
    out[i] = val
  end
end

function filt{T,S,G}(b::T, a::PolyMatrix{S},
  x::AbstractArray{G}, si=zeros(promote_type(S, G, T), order(a), size(a,1)))
  out = Array(promote_type(T, G, S), size(x,1), size(x,2))
  _filt_ar!(out, a, x, si)  # TODO should be a/b
  return out
end

function _filt_ar!{T,S}(
  out::AbstractMatrix{T}, a::PolyMatrix{T},
  x::AbstractArray{T}, si::AbstractMatrix{S})
  silen = size(si,1)
  ac = coeffs(a)
  val = zeros(T,1,size(a,2))
  @inbounds @simd for i=1:size(x, 1)
    xi = view(x,i,:)
    val = si[1,:] + xi
    for j=1:(silen-1)
       si[j,:] = si[j+1,:,] - ac[j]*val
    end
    si[silen,:] = - ac[silen]*val
    out[i,:] = val
  end
end

# function _filt_iir!{M<:AbstractArray,T,S,N}(
#   out::AbstractArray, b::Vector{M}, a::Vector{M},
#   x::AbstractArray{T}, si::AbstractArray{S,N})
#   silen = size(si,2)
#   @inbounds @simd for i=1:size(x, 2)
#     xi = view(x,:,i)
#     val = si[:,1] + b[1]*xi
#     for j=1:(silen-1)
#       si[:,j] = si[:,j+1] + b[j+1]*xi - a[j+1]*val
#     end
#     si[:,silen] = b[silen+1]*xi - a[silen+1]*val
#     out[:,i] = val
#   end
# end
#
# function _filt_fir!{M<:AbstractArray,T,S,N}(
#   out::AbstractArray, b::Vector{M},
#   x::AbstractArray{T}, si::AbstractArray{S,N})
#   silen = size(si,2)
#   @inbounds @simd for i=1:size(x, 2)
#     xi = view(x,:,i)
#     val = si[:,1] + b[1]*xi
#     for j=1:(silen-1)
#       si[:,j] = si[:,j+1] + b[j+1]*xi
#     end
#     si[:,silen] = b[silen+1]*xi
#     out[:,i] = val
#   end
# end
#
# function _filt_iir!{T}(
#   out::AbstractMatrix{T}, b::AbstractMatrix{T}, a::AbstractMatrix{T},
#   x::AbstractMatrix{T}, si::AbstractMatrix{T})
#   silen = size(si,1)
#   ny = size(out,2)
#   @inbounds @simd for i=1:size(x, 1)
#     xi = view(x,i,:)
#     val = si[1,:] + b[1:ny,:]*xi
#     for j=1:(silen-1)
#       si[j,:] = si[:,j+1] + b[j*ny+(1:ny),:]*xi - a[j*ny+(1:ny),:]*val
#     end
#     si[:,silen] = b[(silen+1)*ny+(1:ny),:]*xi - a[(silen+1)*ny+(1:ny),:]*val
#     out[i,:] = val
#   end
# end
#
# function _filt_fir!{M<:AbstractArray,T,S,N}(
#   out::AbstractMatrix{T}, b::AbstractMatrix{T},
#   x::AbstractMatrix{T}, si::AbstractMatrix{T})
#   silen = size(si,2)
#   @inbounds @simd for i=1:size(x, 2)
#     xi = view(x,:,i)
#     val = si[:,1] + b[1]*xi
#     for j=1:(silen-1)
#       si[:,j] = si[:,j+1] + b[j+1]*xi
#     end
#     si[:,silen] = b[silen+1]*xi
#     out[:,i] = val
#   end
# end
