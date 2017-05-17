# Returns the value function V and saves the gradient/hessian in `storage` as storage = [H g].
function gradhessian!{T<:Real,T2<:Real,S,U,M,O}(
    data::IdDataObject{T}, model::PolyModel{S,U,M}, x::Vector{T2},
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2}, options::IdOptions{O}=IdOptions())
  # check if this is a new point
  if x != last_x
    # update last_x
    copy!(last_x, x)

    y  = data.y
    ny = data.ny
    N  = size(y,1)

    Psit  = psit(data, model, x, options)  # Psi the same for all outputs
    y_est = predict(data, model, x, options)
    eps   = y - y_est
    V     = cost(y, y_est, N, options)

    k = size(Psit,2)

    gt = zeros(T,1,k)
    H  = zeros(T,k,k)

    A_mul_B!(H,  Psit.', Psit)          # H = Psi*Psi.'
    for i = 0:ny-1
      A_mul_B!(gt, -eps[i+1,:].', Psit)   # g = -Psi*eps
      storage[i*k+(1:k), ny*k+1]    = gt.'/N
      storage[i*k+(1:k), i*k+(1:k)] = H/N
    end

    # update last_V
    copy!(last_V, V)

    return V
  end
  return last_V[1]
end

function psit{T<:Real,V1,V2,O,M}(
    data::IdDataObject{T,V1,V2},
    model::PolyModel{Val{:siso},
    FullPolyOrder{Val{:siso}},M},
    Θ::Vector{T}, options::IdOptions{O}=IdOptions())

  estimate_initial   = options.estimate_initial
  na,nb,nf,nc,nd,nk  = orders(model)

  ny,nu     = data.ny, data.nu
  y,u,N     = data.y, data.u, data.N
  k         = na+nb+nf+nc+nd

  Θₚ,icbf,icdc,iccda = _split_params(model, Θ, options)

  a,b,f,c,d = _getpolys(model, Θₚ)
  onepoly   = PolyMatrix(ones(T,1,1), :z̄)

  nbf       = max(nb, nf)
  ndc       = max(nd, nc)
  ncda      = max(nc, nd+na)
  Psit      = estimate_initial ? zeros(T,N,k+nbf+ndc+ncda) : zeros(T,N,k)

  w         = filt(b, f, u, icbf)
  y_est     = predict(data, model, Θ, options)
  ϵ         = y - y_est
  v         = w - filt(onepoly, a, y)

  cump = 0
  # a
  _fill_psi(Psit, N, cump, na, d, c, y)
  cump += na
  # b
  _fill_psi(Psit, N, cump, nb,  d, c*f, u)
  cump += nb
  # f
  _fill_psi(Psit, N, cump, nf, -d, c*f, w)
  cump += nf
  # c
  _fill_psi(Psit, N, cump, nc, -onepoly, c, ϵ)
  cump += nc
  # d
  _fill_psi(Psit, N, cump, nd, onepoly, c, v)
  cump += nd

  if estimate_initial
    _fill_psi_ic(Psit, N, cump, nbf, b, f)
    cump += nbf
    _fill_psi_ic(Psit, N, cump, ndc, d, c)
    cump += ndc
    _fill_psi_ic(Psit, N, cump, ncda, c-d*a, c)
    cump += ncda
  end

  return Psit
end

function _fill_psi_ic(Psit, N, m, n, a, b)
  if n > 0
    T        = eltype(Psit)
    state    = zeros(T, 1, n)
    state[1] = one(T)
    v        = filt(a, b, zeros(T,1,N), state)
    Psit[1:N,m+(1:n)] = Toeplitz(reshape(v,N,1), hcat(v[1],zeros(T,1,n-1)))
  end
end

function _fill_psi(Psit, N, m, n, a, b, u)
  if n > 0
    v = filt(a,b,u)
    T = eltype(v)
    col               = vcat(zeros(T,1,1), v[:,1:end-1].')
    Psit[1:N,m+(1:n)] = Toeplitz(col, zeros(T,1,n))
  end
end
