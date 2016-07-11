# IterativeIdMethod definition

immutable BJ <: IterativeIdMethod
  ic::Symbol
  autodiff::Bool

  @compat function (::Type{BJ})(ic::Symbol, autodiff::Bool)
      new(ic, autodiff)
  end
end

function BJ(; ic::Symbol=:truncate, autodiff::Bool = false)
  BJ(ic, autodiff)
end

function fval{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::BJ,
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  calc_bj(data, n, x, method)
end

function gradhessian!{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::BJ,
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  calc_bj_der!(data, n, x, method, last_x, last_V, storage)
end

function IdDSisoRational{T<:Real}(data::IdDataObject{T}, n::Vector{Int},
    opt::Optim.OptimizationResults, method::BJ)
  _bj(data, n, opt, method)
end

"""
    `bj(data, nb, nc, nd, nf, nk=1)`

Compute the Box-Jenkins (`nb`,`nc`,`nd`,`nf`,`nk`) model:
    y(t) = z^-`nk`B(z)/F(z)u(t) + C(z)/D(z)e(t)
that minimizes the mean square one-step-ahead prediction error for time-domain IdData `data`.

An initial parameter guess can be provided by adding `x0 = [B; C; D; F]` to the argument list,
where B`, `C`, `F` and `F` are vectors.

To use automatic differentiation add `autodiff=true`.
"""
function bj{T<:Real}(data::IdDataObject{T}, nb::Int, nc::Int, nd::Int, nf::Int, nk::Int=1,
    x0::AbstractArray = vcat(init_cond(data.y, data.u, nf, nb, nc, nd)...); kwargs...)
  nfc = nc + nf
  nfd = nd + nf
  ndb = nd + nk + nb - 1
  N   = size(data.y,1)
  m   = max(nc, nfc, nfd, ndb)+1
  n   = [nb, nc, nd, nf, nk]
  k   = sum(n[1:4])

  # detect input errors
  any(n .< 0)     && error("na, nb, nc, nd, nk must be nonnegative integers")
  m>N             && error("Not enough datapoints to fit BJ($na,$nb,$nc,$nd,$nk) model")
  length(x0) != k && error("Used initial guess of length $(length(x0)) for BJ model with $k parameters")

  pem(data, n, x0, BJ(kwargs...))
end

function bj{T<:Real}(data::IdDataObject{T}, n::Vector{Int},
    x0::AbstractArray= vcat(init_cond(data.y, data.u, 1, 1, 1)...); kwargs...)
  bj(data, n..., x0; kwargs...)
end

# create model from optimization data

function _bj{T<:Real}(data::IdDataObject{T}, n::Vector{Int},
    opt::Optim.OptimizationResults, method::BJ)
  nb,nc,nd,nf,nk = n
  nfc            = nc + nf
  nfd            = nd + nf
  ndb            = nd + nk + nb - 1
  N              = size(data.y,1)
  m              = max(nc, nfc, nfd, ndb)+1

  # extract results from opt
  x        = opt.minimum
  mse      = opt.f_minimum
  modelfit = 100 * (1 - sqrt((N-m)*mse) / norm(data.y[m:N]-mean(data.y[m:N])))

  a,b,c,d,f  = _getvec(n, x, method)
  info = IterativeIdInfo(mse, modelfit, opt, method, n)
  IdDSisoRational(a, b, c, d, f, data.Ts, info)
end

function _getvec{T<:Real}(n::AbstractVector{Int}, x::AbstractVector{T}, method::BJ)
  nb,nc,nd,nf,nk = n
  a = ones(T,1)
  b = vcat(zeros(T,nk), x[1:nb])
  c = vcat(ones(T,1), x[nb+1:nb+nc])
  d = vcat(ones(T,1), x[nb+nc+1:nb+nc+nd])
  f = vcat(ones(T,1), x[nb+nc+nd+1:end])
  return a,b,c,d,f
end

function predict{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::BJ)
  y, u = data.y, data.u
  T    = promote_type(T1, T2)
  ic   = method.ic

  nb, nc, nd, nf, nk = n
  nfc                = nc + nf
  nfd                = nd + nf
  ndb                = nd + nk + nb - 1
  @assert length(x) == sum(nb+nc+nd+nf) string("length of parameter vector does not match orders in n")

  mcd = max(nc,nd)
  m = max(nc, nfc, nfd, ndb)+1
  b = vcat(zeros(T,nk), x[1:nb])
  c = vcat(ones(T,1), x[nb+1:nb+nc], zeros(T,m-nc))
  d = vcat(ones(T,1), x[nb+nc+1:nb+nc+nd], zeros(T,m-nd))
  f = vcat(ones(T,1), x[nb+nc+nd+1:nb+nc+nd+nf])

  # calculate products of polynomials and zero-pad
  bd = vcat(conv(b,d)::Vector{T}, zeros(T,m-1-ndb)) # conv has problematic return type
  fd = vcat(conv(f,d)::Vector{T}, zeros(T,m-1-nfd))
  fc = vcat(conv(f,c)::Vector{T}, zeros(T,m-1-nfc))

  N     = length(y)
  y_est = zeros(T,N)
  if ic == :truncate
    # assume y_est = y for t=1:m to find initial states for the filters
    y_est[1:m]   = y[1:m]
    si           = filtic(bd, fc, y[m:-1:1]/2, u[m:-1:1])    # TODO not correct,
    si2          = filtic(c-d, c, y[m:-1:1]/2, y[m:-1:1])    # fix at some point by including in optimization
    y_est[m+1:N] = filt(bd, fc, u[m+1:N], si) + filt(c-d, c, y[m+1:N], si2)
  else
    # zero initial conditions
    y_est        = filt(bd, fc, u) + filt(c-d, c, y)
  end
  return y_est
end

# calculate the value function V. Used for automatic differentiation
function calc_bj{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::BJ)
  nb, nc, nd, nf, nk = n
  nfc                = nc + nf
  nfd                = nd + nf
  ndb                = nd + nk + nb - 1
  m                  = max(nc, nfc, nfd, ndb)+1

  y, u = data.y, data.u
  ic   = method.ic

  N     = length(y)
  y_est = predict(data, n, x, method)
  if ic == :truncate
    return sumabs2(y-y_est)/(N-m)
  else
    return sumabs2(y-y_est)/N
  end
end

# Returns the value function V and saves the gradient/hessian in `storage` as storage = [H g].
function calc_bj_der!{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, T2<:Real}(
  data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::Vector{T2}, method::BJ,
  last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2})
  # check if this is a new point
  if x != last_x
    # update last_x
    copy!(last_x, x)
    y, u = data.y, data.u
    ic   = method.ic

    nb,nc,nd,nf,nk = n
    nfc            = nc + nf
    nfd            = nd + nf
    ndb            = nd + nk + nb - 1
    m              = max(nc, nfc, nfd, ndb)+1
    k              = nf+nb+nc+nd
    N              = length(y)

    b = vcat(zeros(T2,nk), x[1:nb])
    c = vcat(ones(T2,1), x[nb+1:nb+nc])
    d = vcat(ones(T2,1), x[nb+nc+1:nb+nc+nd])
    f = vcat(ones(T2,1), x[nb+nc+nd+1:end])

    # calculate products of polynomials and zero-pad
    bd = vcat(conv(b,d)::Vector{T2}, zeros(T2,m-1-ndb)) # conv has problematic return type
    fd = vcat(conv(f,d)::Vector{T2}, zeros(T2,m-1-nfd))
    fc = vcat(conv(f,c)::Vector{T2}, zeros(T2,m-1-nfc))

    y_est = predict(data, n, x, method)
    if ic == :truncate
      V = sumabs2(y-y_est)/(N-m)
    else
      V = sumabs2(y-y_est)/N
    end

    w    = filt(b,f,y)
    eps  = y - y_est
    v    = y - w
    uf   = filt(d, fc, u)
    wf   = filt(d, fc, w)
    epsf = filt(ones(T2,1), c, eps)
    vf   = filt(ones(T2,1), c, v)

    Psit = zeros(T2,N,k) # set all initial conditions on the derivative to zero (not sure if this is correct..)

    @simd for i = 1:nb
      @inbounds Psit[m:N, i]         = uf[m-nk+1-i:N-nk+1-i]
    end
    @simd for i = 1:nc
      @inbounds Psit[m:N,nb+i]       = epsf[m-i:N-i]
    end
    @simd for i = 1:nd
      @inbounds Psit[m:N,nb+nc+i]    = -vf[m-i:N-i]
    end
    @simd for i = 1:nf
      @inbounds Psit[m:N,nb+nc+nd+i] = wf[m-i:N-i]
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

#=   Function: init_cond

Calculates intitial conditions for PEM.

Author : Cristian Rojas, Linus H-Nielsen
=#
function init_cond(y, u, na, nb, nc, nd=0)
    # TODO: add checks to make sure there are enough samples
    trans = 50
    nf = 3*nc + 6*nd
    nl = 6*nc

    # TODO: allow for time delayed input
    nk = 1

    N = length(u)
    m = max(na, nb+nk-1) + 1

    # compute arx model ignoring noise terms
    theta, phi = arx_(y, u, na, nb, nk)

    # generate instrumental variables
    z = generate_instruments(theta, u, na, nb)[:,m:end-1]

    # compute instrumental variables estimate
    theta = (z*phi) \ z*y[m:end]

    if nd >= 1
        # Obtain residuals v(t) = C/D*e(t) = y(t) - B/A*u(t)
        v   = zeros(N)
        tmp = zeros(na+nb)
        vtmp = zeros(na)
        for i = 1:N
            v[i] = y[i] + dot(tmp, theta) + dot(vtmp, theta[1:na])
            tmp = [y[i]; tmp[1:na-1]; -u[i]; tmp[na+1:na+nb-1]]
            vtmp = [-v[i]; vtmp[1:na-1]]
        end

        # approximate the residuals as a high order AR process
        F = ar_(v, nf)

        # Compute e(t) = v(t) / F
        e   = zeros(N)
        tmp = zeros(nf)

        for i = 1:N
            e[i] = v[i] + dot(tmp, F)
            tmp  = [-e[i]; tmp[1:nf-1]]
        end

        # estimate noise parameters
        theta2 = arx_(v, e, nd, nc, 1)[1]

        # Compute v₂(t) = C/D*e(t)
        tmp = zeros(nc+nd)
        for i = 1:N
            v[i] = e[i] + dot(tmp, theta2)
            tmp = [-v[i]; tmp[1:nd-1]; e[i]; tmp[nd+1:nd+nc-1]]
        end

        theta = arx_(u, y-v, nb, na, 1)[1]
        A = theta[1:na]
        B = theta[na+1:end]
        C = theta2[nd+1:end]
        D = theta2[1:nd]
    else
        # v(t) = C*e(t) = A*y(t) - B*u(t)
        tmp = zeros(na+nb)
        v = zeros(N)
        for i = 1:N
            v[i] = y[i] + dot(tmp, theta)
            tmp  = [y[i]; tmp[1:na-1]; -u[i]; tmp[na+1:na+nb-1]]
        end

        # Estimate L=1/C as a high order AR process
        L = ar_(v, nl)

        # Obtain e(t) = v(t) / C
        e   = zeros(N)
        tmp = zeros(nl)
        for i = 1:N
            e[i] = v[i] + dot(tmp, L)
            tmp  = [-e[i]; tmp[1:nl-1]]
        end

        # Estimate A, B, C from y, u and e
        if nc >= 1
            phi    = zeros(na+nb+nc,1)
            theta  = zeros(na+nb+nc,1)
            P      = 10*var(y)*eye(na+nb+nc)

            for i = trans+1:N
                temp  = phi'*theta
                r     = y[i] - temp[1]
                temp  = 1 + phi'*P*phi
                P     = P - P*phi*phi'*P/temp[1]
                theta = theta + P*phi*r
                phi   = [-y[i]; phi[1:na-1]; u[i]; phi[na+1:na+nb-1]; e[i]; phi[na+nb+1:na+nb+nc-1]]
            end

            A = collect(theta[1:na])
            B = theta[na+1:na+nb]
            C = collect(theta[na+nb+1:na+nb+nc])
            D = Float64[]

        else
            phi    = zeros(na+nb,1)
            theta  = zeros(na+nb,1)
            P      = 10*var(y)*eye(na+nb)
            for i = trans+1:N
                temp  = phi'*theta
                r     = y[i] - temp[1]
                temp  = 1 + phi'*P*phi
                P     = P - P*phi*phi'*P/temp[1]
                theta = theta + P*phi*r
                phi   = [-y[i]; phi[1:na-1]; u[i]; phi[na+1:na+nb-1]]
            end

            A = collect(theta[1:na])
            B = theta[na+1:na+nb]
            C = Float64[]
            D = Float64[]
        end
    end
    return A,B,C,D
end

function arx_(y, u, na, nb, nk)
    # Number of samples
    N = length(y)

    # Time horizon
    m = max(na, nb+nk-1)+1

    # Number of parameters
    n = na + nb

    # Estimate parameters
    Phi = Matrix{Float64}(N-m+1, n)
    for i=m:N
        Phi[i-m+1,:] = hcat(-y[i-1:-1:i-na]', u[i-nk:-1:i-nk-nb+1]')
    end
    theta = Phi\y[m:N]
    return theta, Phi
end

function generate_instruments(theta, u, na, nb)
    N = length(u)
    m = max(na, nb+nb-1) + 1
    # generate instruments
    z = zeros(na+nb, N+1)
    for i=1:N
        z[:,i+1] = [-dot(z[:,i],theta); z[1:na-1,i]; u[i]; z[na+1:na+nb-1,i]]
    end
    return z
end

# models v as an AR process: A*v ≈ e
function ar_(v, nk)
    theta  = zeros(nk,1)
    phi    = zeros(nk,1)
    P      = 10*var(v)*eye(nk)
    for i = 1:length(v)
        e     = v[i] - phi'*theta
        temp  = 1 + phi'*P*phi
        P     = P - P*phi*phi'*P/temp[1]
        theta = theta + P*phi*e
        phi   = [-v[i]; phi[1:nk-1]]
    end
    return collect(theta)
end
