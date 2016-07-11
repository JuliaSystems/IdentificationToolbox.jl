
immutable STMCB <: OneStepIdMethod
  ic::Symbol
  feedthrough::Bool
  maxiter::Int
  tol::Float64

  @compat function (::Type{STMCB})(ic::Symbol, feedthrough::Bool, maxiter::Int, tol::Float64)
    @assert in(ic, Set([:truncate,:zero]))  string("ic must be either :truncate or :zero")
    @assert maxiter > 0                     string("maxiter need to be greater than zero")
    @assert tol >= 0                        string("tol need to be greater or equal to zero")

    new(ic, feedthrough, maxiter, tol)::STMCB
  end
end

function STMCB(; ic::Symbol=:zero, feedthrough::Bool=false, maxiter::Int=20, tol::Float64=1e-10)
  STMCB(ic, feedthrough, maxiter, tol)
end

function fitmodel{T<:Real}(data::IdDataObject{T}, n::Vector{Int}, method::STMCB; kwargs...)
  stmcb(data, n, method)
end

function stmcb{T<:Real}(data::IdDataObject{T}, n::Vector{Int}, method::STMCB=STMCB())
  nb,nf,nk = n
  m        = max(nf, nb+nk-1)+1
  N        = data.N
  @assert nf    > -1 string("nf must be positive")
  @assert nb    > -1 string("nb must be positive")
  @assert nk    > -1 string("nk must be positive")
  @assert nf+nb >  0 string("nb+nf must be greater than zero")

  x, mse = _stmcb(data, n, method)
  modelfit = 100 * (1 - sqrt((N-m)*mse) / norm(data.y[m:N]-mean(data.y[m:N])))

  a,b,c,d,f = _getvec(n, x, method)
  info      = OneStepIdInfo(mse, modelfit, method, n)
  IdDSisoRational(a, b, c, d, f, data.Ts, info)
end

function _getvec{T<:Real}(n::AbstractVector{Int}, x::AbstractVector{T}, method::STMCB)
  nb,nf,nk = n
  if method.feedthrough
    b = vcat(ones(T,1), zeros(T,nk-1), x[1:nb])
  else
    b = vcat(zeros(T,nk), x[1:nb])
  end
  a = ones(T,1)
  c = ones(T,1)
  d = ones(T,1)
  f = vcat(ones(T,1), x[nb+1:end])
  return a,b,c,d,f
end

function _stmcb{T<:Real, V1<:AbstractVector, V2<:AbstractVector}(
    data::IdDataObject{T,V1,V2}, n::Vector{Int}, method::STMCB=STMCB(),
    c::Vector{T}=ones(T,0), d::Vector{T}=ones(T,0))
  nb, nf, nk  = n
  m           = nb+nf
  y, u        = data.y, data.u
  nc          = length(c)
  nd          = length(d)
  feedthrough = method.feedthrough
  ic          = method.ic
  maxiter     = method.maxiter
  tol         = method.tol
  # StMcB computes a model using the Steiglitz-McBride method
@assert nb >= 0 && nf >= 0 string("nb and nf must be larger or equal to zero")
@assert nk >= 0 string("nk must be greater or equal to zero")
@assert !feedthrough || nk > zero(Int) string("nk must be greater than zero if feedthrough term is known")

  # first iteration the data is not pre-filtered
  yf     = copy(data.y)
  uf     = copy(data.u)
  dataf  = iddata(yf, uf, data.Ts)

  bestpe = typemax(Float64)
  best_b = zeros(T,1)
  best_f = ones(T,1)
  Θ      = zeros(m)
  Θp     = zeros(m)
  for i = 1:maxiter
    Θ = _arx(dataf, [nf,nb,nk], ARX(ic, feedthrough))[1]
    f = Θ[1:nf]
    b = Θ[nf+1:end]
    if feedthrough
      x  = vcat(ones(T,1), zeros(T,nk-1), [b; c; d; f])
      pe = calc_bj(data, [nb+nk,nc,nd,nf,0], x, BJ(ic=ic))
    else
      x  = vcat(b,c,d,f)
      pe = calc_bj(data, [nb,nc,nd,nf,nk], x, BJ(ic=ic))
    end

    if pe < bestpe
      best_b = b
      best_f = f
      bestpe = pe
    end
  #  if norm(Θ-Θp) < norm(Θp)*tol
  #    return vcat(best_b, best_f), bestpe
  #  end

    # filter data
    F     = vcat(ones(T,1), f)
    filt!(yf,one(T), F, y)
    filt!(uf,one(T), F, u)
    dataf = iddata(yf, uf, data.Ts)
    Θp    = Θ
  end
  return vcat(best_b, best_f), bestpe
end
