# IterativeIdMethod definition

immutable ARMAX <: IterativeIdMethod
  ic::Symbol
  autodiff::Bool

  @compat function (::Type{ARMAX})(ic::Symbol, autodiff::Bool)
    @assert in(ic, Set([:truncate,:zero]))  string("ic must be either :truncate or :zero")
      new(ic, autodiff)
  end
end

function ARMAX(;ic::Symbol=:truncate, autodiff::Bool=false)
  ARMAX(ic, autodiff)
end

function fval{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::ARMAX,
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  calc_armax(data, n, x, method)
end

function gradhessian!{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::ARMAX,
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  calc_armax_der!(data, n, x, method, last_x, last_V, storage)
end

function IdDSisoRational{T<:Real}(data::IdDataObject{T}, n::Vector{Int},
    opt::Optim.OptimizationResults, method::ARMAX)
  _armax(data, n, opt, method)
end

"""
    `armax(data, na, nb, nc, nk=1)`

Compute the ARMAX(`na`,`nb`,`nc`,`nd`) model:
    A(z)y(t) = z^-`nk`B(z)u(t) + C(z)e(t)
that minimizes the mean square one-step-ahead prediction error for time-domain IdData `data`.

An initial parameter guess can be provided by adding `x0 = [A; B; C]` to the argument list, where `A`, `B` and `C` are vectors.

To use automatic differentiation add `autodiff=true`.
"""
function armax{T1<:Real,V1<:AbstractVector,V2<:AbstractVector}(
    data::IdDataObject{T1,V1,V2}, na::Int, nb::Int, nc::Int, nk::Int=1,
    x0::AbstractArray = vcat(init_cond(data.y, data.u, na, nb, nc)...); kwargs...)
  N = size(data.y, 1)
  m = max(na, nb+nk-1, nc)+1
  n = [na,nb,nc,nk]
  k = na + nb + nc

  # detect input errors
  any(n .< 0)     && error("na, nb, nc, nk must be nonnegative integers")
  m>N             && error("Not enough datapoints to fit ARMAX($na,$nb,$nc,$nk) model")
  length(x0) != k && error("Used initial guess of length $(length(x0)) for ARMAX model with $k parameters")

  pem(data, n, x0, ARMAX(kwargs...))
end

function armax{T1<:Real,V1<:AbstractVector,V2<:AbstractVector}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int},
    x0::AbstractArray = vcat(init_cond(data.y, data.u, n[1], n[2], n[3])...); kwargs...)
  armax(data, n..., x0; kwargs...)
end

# create model from optimization data

function _armax{T<:Real}(data::IdDataObject{T}, n::Vector{Int},
    opt::Optim.OptimizationResults, method::ARMAX)
  na,nb,nc,nk = n
  N = size(data.y, 1)
  m = max(na, nb+nk-1, nc)+1
  k = na + nb + nc

  # extract results from opt
  x = opt.minimum
  mse = opt.f_minimum
  modelfit = 100 * (1 - sqrt((N-m)*mse) / norm(data.y[m:N]-mean(data.y[m:N])))

  a = vcat(ones(T,1),   x[1:na])
  b = vcat(zeros(T,nk), x[na+1:na+nb])
  c = vcat(ones(T,1),   x[na+nb+1:end])
  d = ones(T,1)
  f = ones(T,1)
  info = IterativeIdInfo(mse, modelfit, opt, method, n)

  IdDSisoRational(a, b, c, d, f, data.Ts, info)
end

function predict{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
  data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::ARMAX)
  na, nb, nc, nk = n
  y, u           = data.y, data.u
  T = promote_type(T1, T2)
  ic = method.ic

  m = max(na, nb+nk-1, nc)
  # zero pad vectors
  a = append!(append!(ones(T,1), x[1:na]), zeros(T,m-na))
  b = append!(append!(zeros(T,nk), x[na+1:na+nb]), zeros(T,m+1-nb-nk))
  c = append!(append!(ones(T,1), x[na+nb+1:end]), zeros(T,m-nc))
  N = length(y)
  y_est = zeros(T,N)
  if ic == :truncate
    # assume y_est = y for t=1:m to find initial states for the filters
    y_est[1:m]   = y[1:m]
    si           = filtic(b, c, y[m:-1:1]/2, u[m:-1:1])
    si2          = filtic(c-a, c, y[m:-1:1]/2, y[m:-1:1])
    y_est[m+1:N] = filt(b, c, u[m+1:N], si) + filt(c-a, c, y[m+1:N], si2)
  else
    # zero initial conditions
    y_est = filt(b, c, u) + filt(c-a, c, y)
  end
  return y_est
end

# cost function
function calc_armax{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::ARMAX)
  na, nb, nc, nk = n
  y, u           = data.y, data.u
  ic             = method.ic
  m              = max(na, nb+nk-1, nc)

  N     = length(y)
  y_est = predict(data, n, x, method)
  if ic == :truncate
    return sumabs2(y-y_est)/(N-m)
  else
    return sumabs2(y-y_est)/N
  end
end

# Returns the value function V and saves the gradient/hessian in `storage` as storage = [H g].
function calc_armax_der!{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::ARMAX,
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  # check if this is a new point
  if x != last_x
    copy!(last_x, x)
    y, u = data.y, data.u
    ic   = method.ic

    na, nb, nc, nk = n
    m              = max(na, nb+nk-1, nc)+1
    N              = length(y)
    k              = na+nb+nc

    y_est = predict(data, n, x, method)
    if ic == :truncate
      V = sumabs2(y-y_est)/(N-m)
    else
      V = sumabs2(y-y_est)/N
    end

    c          = append!(append!(ones(T2,1), x[na+nb+1:end]), zeros(T2,m-nc))
    one_size_c = append!(ones(T2,1), zeros(T2,nc-1))
    eps        = y-y_est
    yf         = filt(one_size_c, c, y)
    uf         = filt(one_size_c, c, u)
    epsf       = filt(one_size_c, c, eps)

    Psit = zeros(T2,N,k)
    @simd for i = 1:na
      @inbounds Psit[m:N,i]        = -yf[m-i:N-i]
    end
    @simd for i = 1:nb
      @inbounds Psit[m:N, na+i]    = uf[m-nk+1-i:N-nk+1-i]
    end
    @simd for i = 1:nc
      @inbounds Psit[m:N, na+nb+i] = epsf[m-i:N-i]
    end

    gt = zeros(1,k)
    H  = zeros(k,k)
    A_mul_B!(gt, -eps.', Psit)   # g = -Psi*eps
    A_mul_B!(H,  Psit.', Psit)   # H = Psi*Psi.'
    storage[1:k, k+1] = gt.'
    storage[1:k, 1:k] = H

    # normalize
    storage[1:k, k+1] /= N-m+1
    storage[1:k, 1:k] /= N-m+1

    # update last_V
    copy!(last_V, V)

    return V
  end
  return last_V[1]
end
