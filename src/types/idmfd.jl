immutable IdMFD{S,C,L,M1,M2,M3,M4,M5,C1,C2} <: SystemsBase.LtiSystem{S,C}
  A::M1
  B::M2
  F::M3
  C::M4
  D::M5
  G::C1
  H::C2
  info::IdInfo

  # Discrete-time, single-input-single-output MFD model
  @compat function (::Type{IdMFD}){T}(A::Poly{T}, B::Poly{T}, F::Poly{T}, C::Poly{T},
    D::Poly{T}, Ts::Float64, info::IdInfo{Val{:siso}})
    G = SystemsBase.lfd(B, A*F, Ts)
    H = SystemsBase.lfd(C, A*D, Ts)
    M = Poly{T}
    new{Val{:siso},Val{:disc},Val{:lfd},M,M,M,M,M,typeof(G),typeof(H)}(
      A, B, F, C, D, G, H, info)
  end

  @compat function (::Type{IdMFD}){M1<:PolynomialMatrices.PolyMatrix,
    M2<:PolynomialMatrices.PolyMatrix, M3<:PolynomialMatrices.PolyMatrix,
    M4<:PolynomialMatrices.PolyMatrix, M5<:PolynomialMatrices.PolyMatrix}(
      A::M1, B::M2, F::M3, C::M4, D::M5, Ts::Float64, info::IdInfo{Val{:siso}})
    G = SystemsBase.lfd(B[1], (A*F)[1], Ts)
    H = SystemsBase.lfd(C[1], (A*D)[1], Ts)
    new{Val{:siso},Val{:disc},Val{:lfd},M1,M2,M3,M4,M5,
      typeof(G),typeof(H)}(
      A, B, F, C, D, G, H, info)
  end

  # Discrete-time, multi-input-multi-output MFD model
  @compat function (::Type{IdMFD}){M1<:PolynomialMatrices.PolyMatrix,
    M2<:PolynomialMatrices.PolyMatrix, M3<:PolynomialMatrices.PolyMatrix,
    M4<:PolynomialMatrices.PolyMatrix, M5<:PolynomialMatrices.PolyMatrix}(
      A::M1, B::M2, F::M3, C::M4, D::M5, Ts::Float64, info::IdInfo{Val{:mimo}})
    G = SystemsBase.lfd(B, A*F, Ts)
    H = SystemsBase.lfd(C, A*D, Ts)
    new{Val{:mimo},Val{:disc},Val{:lfd},M1,M2,M3,M4,M5,
      typeof(G),typeof(H)}(
      A, B, F, C, D, G, H, info)
  end
end

# Outer constructors
function IdMFD{T<:AbstractFloat, S}(a::AbstractVector{T},
    b::AbstractVector{T}, f::AbstractVector{T}, c::AbstractVector{T},
    d::AbstractVector{T}, Ts::Float64, info::IdInfo{S})
  na = length(a)
  nb = length(b)
  nf = length(f)
  nc = length(c)
  nd = length(d)
  naf = na + nf - 1
  nad = na + nd - 1

  mG = max(naf, nb)
  mH = max(nad, nc)

  b = vcat(b, zeros(T, mG-nb))
  c = vcat(c, zeros(T, mH-nc))
  d = vcat(d)
  f = vcat(f)

  A = Poly(reverse(a))
  B = Poly(reverse(b))
  F = Poly(reverse(f))
  C = Poly(reverse(c))
  D = Poly(reverse(d))

  IdMFD(A,B,F,C,D,Ts,info)
end

function IdMFD{T<:AbstractFloat, S}(a::AbstractMatrix{T},
    b::AbstractMatrix{T}, f::AbstractMatrix{T}, c::AbstractMatrix{T},
    d::AbstractMatrix{T}, Ts::Float64, info::IdInfo{S}, ny::Int)

  A = PolyMatrix(a, (ny, size(a,2)))
  B = PolyMatrix(b, (ny, size(b,2)))
  F = PolyMatrix(f, (ny, size(f,2)))
  C = PolyMatrix(c, (ny, size(c,2)))
  D = PolyMatrix(d, (ny, size(d,2)))

  IdMFD(A,B,F,C,D,Ts,info)
end

# conversion to tf
tf(s::IdMFD{Val{:siso},Val{:disc}}) = tf(coeffs(s.B[1]), coeffs(s.F[1]), s.G.Ts, :z̄)

lfd(s::IdMFD) = lfd(s.G)

ss(s::IdMFD{Val{:siso},Val{:disc}}) = ss(tf(s))


samplingtime(s::IdMFD) = s.G.Ts
isdiscrete(s::IdMFD)   = true
getmatrix(s::IdMFD)    = reshape([s.G; s.H],1,2)
numstates(s::IdMFD)    = numstates(s.G)
numoutputs(s::IdMFD)   = numoutputs(s.G)
numinputs(s::IdMFD)    = numinputs(s.G)

function _append_params(x, p, p0)
  for (k,v) in coeffs(p)
    if k > p0
      append!(x, v.'[:])
    end
  end
end

function get_params{S,C,L,M1,M2,M3,M4,M5,C1,C2}(
  s::IdMFD{S,C,L,M1,M2,M3,M4,M5,C1,C2})

  T = eltype(mattype(s.A))
  x = Vector{T}(0)
  _append_params(x, s.A, 0)
  _append_params(x, s.B, 0)
  _append_params(x, s.F, 0)
  _append_params(x, s.C, 0)
  _append_params(x, s.D, 0)
  return x
end

# function predict(data::IdDataObject, sys::IdMFD)
#   n = sys.info.n
#   return predict(data, n, get_param(sys,n), sys.info.method)
# end
