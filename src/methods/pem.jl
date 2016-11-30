function fval{T<:IterativeIdMethod, T1<:Real, V1<:AbstractVector,
    V2<:AbstractVector, T2<:Real}(data::IdDataObject{T1,V1,V2},
    n::Vector{Int}, x::Vector{T2}, method::T, last_x::Vector{T2},
    last_V::Vector{T2}, storage::Matrix{T2})
  fval(data, n, x, method, last_x, last_V, storage)
end

function fval{T<:OneStepIdMethod, T1<:Real, V1<:AbstractVector, V2<:AbstractVector}(
  method::T, data::IdDataObject{T1,V1,V2}, n::Vector{Int})
  fval(data, n, x, method)
end

function g!{T<:IterativeIdMethod, T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::T,
    last_x::Vector{T2}, last_V::Vector{T2}, g, storage::Matrix{T2})
    gradhessian!(data, n, x, method, last_x, last_V, storage)
    copy!(g, storage[:, end])
end

function h!{T<:IterativeIdMethod, T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::T,
    last_x::Vector{T2}, last_V::Vector{T2}, H, storage::Matrix{T2})
  gradhessian!(data, n, x, method, last_x, last_V, storage)
  copy!(H, storage[:,1:end-1])
end

function pem{M1<:OneStepIdMethod, T1<:Real}(
    data::IdDataObject{T1}, n::Vector{Int}, method::M1)
  @assert data.ny < 2 string("PEM only implemented for SISO systems")
  @assert data.nu < 2 string("PEM only implemented for SISO systems")

  fitmodel(data, n, method)
end

function pem{M1<:IterativeIdMethod, M2<:OneStepIdMethod, T1<:Real}(
    data::IdDataObject{T1}, n::Vector{Int}, method::M1,
    init_method::M2=MORMSM(version=:H), n_init::Vector{Int}=n)
  @assert data.ny < 2 string("PEM only implemented for SISO systems")
  @assert data.nu < 2 string("PEM only implemented for SISO systems")

  s1 = fitmodel(data, n_init, init_method)
  x0  = get_param(s1,n)
  pem(data, n, x0, method)
end

function pem{M1<:IterativeIdMethod, T1<:Real, V1<:AbstractVector}(
    data::IdDataObject{T1}, n::Vector{Int}, x0::V1, method::M1)
  @assert data.ny < 2 string("PEM only implemented for SISO systems")
  @assert data.nu < 2 string("PEM only implemented for SISO systems")

  k = sum(n[1:end-1]) # number of parameters
  last_x  = zeros(T1,k)
  last_V  = - ones(T1,1)
  autodiff = method.autodiff

  opt::Optim.OptimizationResults
  if !autodiff
    storage = zeros(k, k+1)
    df = TwiceDifferentiableFunction(x    -> fval(data, n, x, method, last_x, last_V, storage),
    (x,g) -> g!(data, n, x, method, last_x, last_V, g, storage),
    (x,H) -> h!(data, n, x, method, last_x, last_V, H, storage))
    opt = optimize(df, x0, Newton(), OptimizationOptions(g_tol = 1e-16))
  else
    opt = optimize(x->f(data, n, x, method, last_x, last_V, storage),
          x0, Newton(), OptimizationOptions(autodiff = true, g_tol = 1e-12))
  end
  IdDSisoRational(data, n, opt, method)
end


# calculate the value function V. Used for automatic differentiation
function cost{T<:Real}(data::IdDataObject{T}, model::PolynomialModel, x)
  #calc_bj{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
  #  data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::BJ)

  y     = data.y
  N     = size(y,1)
  y_est = predict(data, model, x)

  # if ic == :truncate
  #   #TODO proper m
  #   m = 0
  #   return sumabs2(y-y_est)/(N-m)
  # end
  typeof(y)
  return cost(y, y_est, N)
end

cost{T}(y::AbstractArray{T}, y_est::AbstractArray{T}, N::Int) = sumabs2(y-y_est)/N


# Returns the value function V and saves the gradient/hessian in `storage` as storage = [H g].
function gradhessian!{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::BJ,
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  # check if this is a new point
  if x != last_x
    # update last_x
    copy!(last_x, x)

    y = data.y
    N = size(y,1)

    Psit = Psi(data, model, x)
    eps  = y - predict(data, method, x)
    V    = cost(y, y_est, N)

    k = size(Psit,2)

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
