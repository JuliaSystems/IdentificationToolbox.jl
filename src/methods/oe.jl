# IterativeIdMethod definition

immutable OE <: IterativeIdMethod
  ic::Symbol
  autodiff::Bool

  @compat function (::Type{OE})(ic::Symbol, autodiff::Bool)
      new(ic, autodiff)
  end
end

function OE(;ic::Symbol=:truncate, autodiff::Bool=false)
  OE(ic, autodiff)
end

function fval{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::OE,
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  calc_oe(data, n, x, method)
end

function gradhessian!{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::OE,
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  calc_oe_der!(data, n, x, method, last_x, last_V, storage)
end

function IdDSisoRational{T<:Real}(data::IdDataObject{T}, n::Vector{Int},
    opt::Optim.OptimizationResults, method::OE)
  _oe(data, n, opt, method)
end

"""
    `oe(data, nb, nf, nk=1)`

Compute the OE(nb`,`nf`,`nd`) model:
    F(z)y(t) = z^-`nk`B(z)u(t) + F(z)e(t)
that minimizes the mean square one-step-ahead prediction error for time-domain IdData `data`.

An initial parameter guess can be provided by adding `x0 = [B, F]` to the argument list,
where `B` and `F` are vectors.

To use automatic differentiation add `autodiff=true`.
"""
function oe{T<:Real}(data::IdDataObject{T}, nb::Int, nf::Int, nk::Int=1,
      x0::AbstractArray = vcat(init_cond(data.y, data.u, na, nb, nc)...); kwargs...)
  N = size(data.y, 1)
  m = max(nf, nb+nk-1)+1
  n = [nb,nf,nk]
  k = nf + nb

  # detect input errors
  any(n .< 0)     && error("nb, nf, nk must be nonnegative integers")
  m>N             && error("Not enough datapoints to fit OE($nb,$nf,$nk) model")
  length(x0) != k && error("Used initial guess of length $(length(x0)) for OE model with $k parameters")

  pem(data, n, x0, OE(kwargs...))
end

function oe{T<:Real}(data::IdDataObject{T}, n::Vector{Int},
    x0::AbstractArray= vcat(init_cond(data.y, data.u, 1, 1, 1)...); kwargs...)
  oe(data, n..., x0; kwargs...)
end

function _oe{T<:Real}(data::IdDataObject{T}, n::Vector{Int},
    opt::Optim.OptimizationResults, method::OE)
  N        = size(data.y, 1)
  nb,nf,nk = n
  m        = max(nf, nb+nk-1)+1

  # extract results from opt
  x        = opt.minimum
  mse      = opt.f_minimum
  modelfit = 100 * (1 - sqrt((N-m)*mse) / norm(data.y[m:N]-mean(data.y[m:N])))

  a,b,c,d,f = _getvec(n, x, method)
  info      = IterativeIdInfo(mse, modelfit, opt, method, n)
  IdDSisoRational(a, b, c, d, f, data.Ts, info)
end

function _getvec{T<:Real}(n::AbstractVector{Int}, x::AbstractVector{T}, method::OE)
  nb,nf,nk = n
  a = ones(T,1)
  b = vcat(zeros(T,nk), x[1:nb])
  c = ones(T,1)
  d = ones(T,1)
  f = vcat(ones(T,1), x[nb+1:end])
  return a,b,c,d,f
end

function predict{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::OE)
  nb, nf, nk = n
  y, u       = data.y, data.u
  T          = promote_type(T1, T2)
  ic         = method.ic

  m = max(nf, nb+nk-1)
  # zero pad vectors
  b = append!(append!(zeros(T,nk), x[1:nb]), zeros(T,m+1-nb-nk))
  f = append!(append!(ones(T,1), x[nb+1:end]), zeros(T,m-nf))

  N     = length(y)
  y_est = zeros(T,N)
  if ic == :truncate
    # assume y_est = y for t=1:m to find initial states for the filters
    y_est[1:m]   = y[1:m]
    si           = filtic(b, f, y[m:-1:1], u[m:-1:1])
    y_est[m+1:N] = filt(b, f, u[m+1:N], si)
  else
    # zero initial conditions
    y_est = filt(b, f, u)
  end
  return y_est
end

# cost function
function calc_oe{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::OE)
  nb, nf, nk  = n
  y, u        = data.y, data.u
  ic = method.ic
  m           = max(nf, nb+nk-1) + 1

  N           = length(y)
  y_est       = predict(data, n, x, method)
  if ic == :truncate
    return sumabs2(y-y_est)/(N-m)
  else
    return sumabs2(y-y_est)/N
  end
end

# Returns the value function V and saves the gradient/hessian in `storage` as storage = [H g].
function calc_oe_der!{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::OE,
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  # check if this is a new point
  if x != last_x
    # update last_x
    copy!(last_x, x)
    y, u       = data.y, data.u
    ic         = method.ic

    nf, nb, nk = n
    m          = max(nf, nb+nk-1)+1
    N          = length(y)
    k          = nf+nb

    y_est = predict(data, n, x, method)
    if ic == :truncate
      V = sumabs2(y-y_est)/(N-m)
    else
      V = sumabs2(y-y_est)/N
    end

    f          = append!(append!(ones(T2,1), x[nb+1:end]), zeros(T2,m-nf))
    one_size_f = append!(ones(T2,1), zeros(T2,nf-1))
    eps        = y-y_est
    yf         = filt(one_size_f, f, y_est)
    uf         = filt(one_size_f, f, u)

    Psit = zeros(T2,N,k)
    @simd for i = 1:nb
      @inbounds Psit[m:N, i]   = uf[m-nk+1-i:N-nk+1-i]
    end
    @simd for i = 1:nf
      @inbounds Psit[m:N,nb+i] = -yf[m-i:N-i]
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
