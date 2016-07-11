immutable FIR <: OneStepIdMethod
  ic::Symbol

  @compat function (::Type{FIR})(ic::Symbol)
    @assert in(ic, Set([:truncate,:zero]))  string("ic must be either :truncate or :zero")
      new(ic)
  end
end

function FIR(; ic::Symbol=:truncate)
  FIR(ic)
end

function fitmodel{T<:Real}(data::IdDataObject{T}, n::Vector{Int}, method::FIR; kwargs...)
  fir(data, n, method)
end

function fir{T<:Real}(data::IdDataObject{T}, n::Vector{Int}, method::FIR=FIR())
  y,u   = data.y, data.u
  N     = size(y,1)
  nb,nk = n
  m     = nb+nk-1
  @assert nb > 0 "nb must be positive"
  @assert nk > 0 "nk must be positive"

  # estimate model
  x, mse = _fir(data, n, method)

  # Calculate model error
  modelfit = 100*(1 - mse/norm(y[m:N]-mean(y[m:N])))

  a = ones(T,1)
  b = vcat(zeros(T,nk), x[1:nb])
  c = ones(T,1)
  d = ones(T,1)
  f = ones(T,1)
  idinfo = OneStepIdInfo(mse, modelfit, method, n)
  IdDSisoRational(a, b, c, d, f, data.Ts, idinfo)
end

function _fir{T<:Real, V1<:AbstractVector, V2<:AbstractVector}(
    data::IdDataObject{T,V1,V2}, n::Vector{Int}, method::FIR=method(:truncate))
  ic          = method.ic
  y, u        = data.y, data.u
  nb,nk       = n
  N           = size(y,1)
  m           = nb
  md          = nb+nk-1

  theta  = zeros(float(T),nb)
  # Compute efficiently for order m
  col::Array{T,2}
  row::Array{T,2}
  if ic == :truncate
    Y   = y[md+1:N]
    col = reshape(u[m:N-nk],N-md,1)
    row = reverse(u[1:m]).'
  elseif ic == :zero
    Y   = y[1+nk:N]
    col = reshape(u[1:N-nk],N-nk,1)
    row = zeros(T,1,nb)
    row[1,1] = u[1]
  end
  Θ = _ls_toeplitz(col, row, Y)

  # calculate mse
  y_est = _predict_fir(y, u, nb, nk, Θ; ic=ic)
  mse   = sumabs2(data.y-y_est)/(N-m+1)
  return Θ, mse
end

function predict{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, V3<:AbstractVector}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::V3, method::FIR)
  return _predict_fir(data.y, data.u, n..., x; ic=method.ic)
end

function _predict_fir{V1<:AbstractVector, V2<:AbstractVector, V3<:AbstractVector}(
    y::V1, u::V2, nb::Int, nk::Int, theta::V3; ic::Symbol=:truncate)
  T = promote_type(eltype(y), eltype(u), eltype(theta))
  m = nb+nk-1

  b = append!(zeros(T,nk), theta)
  N = length(y)
  y_est = zeros(T,N)
  if ic == :truncate
    # assume y_est = y for t=1:m to find initial states for the filters
    y_est[1:m]   = y[1:m]
    si           = filtic(b,  ones(T,1), filt(b,  ones(T,1), u[m:-1:1]), u[m:-1:1])
    y_est[m+1:N] = filt(b, ones(T,1), u[m+1:N], si)
  else  # ic == :zero
    # zero initial conditions
    y_est = filt(b, ones(T,1), u)
  end
  return y_est
end
