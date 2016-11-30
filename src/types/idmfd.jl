
immutable IdMFD{T,S,M1,M2,M3,M4,M5,C1,C2} <: ControlCore.LtiSystem{T,S}
  A::M1
  B::M2
  C::M3
  D::M4
  F::M5
  G::C1
  H::C2
  info::IdInfo

  # Discrete-time, single-input-single-output MFD model
  @compat function (::Type{IdMFD}){T}(A::Poly{T}, B::Poly{T}, C::Poly{T},
    D::Poly{T}, F::Poly{T}, Ts::Float64, info::IdInfo)
    G = ControlCore.lfd(B, A*F, Ts)
    H = ControlCore.lfd(C, A*D, Ts)
    M = Poly{T}
    new{ControlCore.Siso{true},ControlCore.Continuous{false},M,M,M,M,M,typeof(G),typeof(H)}(
      A, B, C, D, F, G, H, info)
  end

  # Discrete-time, multi-input-multi-output MFD model
  @compat function (::Type{IdMFD}){M1<:PolynomialMatrices.PolyMatrix,
    M2<:PolynomialMatrices.PolyMatrix, M3<:PolynomialMatrices.PolyMatrix,
    M4<:PolynomialMatrices.PolyMatrix, M5<:PolynomialMatrices.PolyMatrix}(
      A::M1, B::M2, C::M3, D::M4, F::M5, Ts::Float64, info::IdInfo)
    G = ControlCore.lfd(B, A*F, Ts)
    H = ControlCore.lfd(C, A*D, Ts)
    new{ControlCore.Siso{false},ControlCore.Continuous{false},M1,M2,M3,M4,M5,typeof(G),typeof(H)}(
      A, B, C, D, F, G, H, info)
  end
end

# Outer constructors
function IdMFD{T<:AbstractFloat, S<:IdInfo}(a::AbstractVector{T},
    b::AbstractVector{T}, c::AbstractVector{T}, d::AbstractVector{T},
    f::AbstractVector{T}, Ts::Float64, info::S)
  na = length(a)
  nb = length(b)
  nc = length(c)
  nd = length(d)
  nf = length(f)
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
  C = Poly(reverse(c))
  D = Poly(reverse(d))
  F = Poly(reverse(f))

  IdMFD(A,B,C,D,F,Ts,info)
end

function IdMFD{T<:AbstractFloat, S<:IdInfo}(a::AbstractMatrix{T},
    b::AbstractMatrix{T}, c::AbstractMatrix{T}, d::AbstractMatrix{T},
    f::AbstractMatrix{T}, Ts::Float64, info::S, ny::Int)

  A = PolyMatrix(a, (ny, size(a,2)))
  B = PolyMatrix(b, (ny, size(b,2)))
  C = PolyMatrix(c, (ny, size(c,2)))
  D = PolyMatrix(d, (ny, size(d,2)))
  F = PolyMatrix(f, (ny, size(f,2)))

  IdMFD(A,B,C,D,F,Ts,info)
end

samplingtime(s::IdMFD) = s.G.Ts
isdiscrete(s::IdMFD)   = true
getmatrix(s::IdMFD)    = reshape([s.G; s.H],1,2)
numstates(s::IdMFD)    = numstates(s.G)
numoutputs(s::IdMFD)   = numoutputs(s.G)
numinputs(s::IdMFD)    = numinputs(s.G)


function get_param{T<:AbstractFloat, S<:IdInfo}(s::IdMFD{T,S},
    n::Vector{Int})
  nk = n[end]
  a = coeffs(s.A)
  b = coeffs(s.B)
  c = coeffs(s.C)
  d = coeffs(s.D)
  f = coeffs(s.F)
  x = Vector{T}(0)
  x = length(a) > 1 ? vcat(x, reverse(a[1:end-1])) : x
  x = vcat(x, reverse(b))
  x = length(c) > 1 ? vcat(x, reverse(c[1:end-1])) : x
  x = length(d) > 1 ? vcat(x, reverse(d[1:end-1])) : x
  x = length(f) > 1 ? vcat(x, reverse(f[1:end-1])) : x
end

function predict{T1<:Real, V1<:AbstractVector, V2<:AbstractVector,
    T2<:AbstractFloat, S<:IdInfo}(
    data::IdDataObject{T1,V1,V2}, sys::IdMFD{T2,S})
  n = sys.info.n
  return predict(data, n, get_param(sys,n), sys.info.method)
end