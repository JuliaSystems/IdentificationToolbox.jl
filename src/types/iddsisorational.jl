immutable IdDSisoRational{T<:AbstractFloat, S<:IdInfo} <: ControlCore.MimoSystem
  A::Poly{T}
  B::Poly{T}
  C::Poly{T}
  D::Poly{T}
  F::Poly{T}
  G::ControlCore.DSisoRational{T,T}
  H::ControlCore.DSisoRational{T,T}
  info::S

  @compat function (::Type{IdDSisoRational}){T,S}(A::Poly{T}, B::Poly{T}, C::Poly{T},
    D::Poly{T}, F::Poly{T}, Ts::Float64, info::S)
    G = ControlCore.tf(B, A*F, Ts)
    H = ControlCore.tf(C, A*D, Ts)
    new{T,S}(A, B, C, D, F, G, H, info)
  end
end

function IdDSisoRational{T<:AbstractFloat, S<:IdInfo}(a::AbstractVector{T},
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

  IdDSisoRational(A,B,C,D,F,Ts,info)
end

ControlCore.samplingtime(s::IdDSisoRational) = s.G.Ts
ControlCore.isdiscrete(s::IdDSisoRational)   = true
ControlCore.getmatrix(s::IdDSisoRational)    = reshape([s.G; s.H],1,2)

function get_param{T<:AbstractFloat, S<:IdInfo}(s::IdDSisoRational{T,S},
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
    data::IdDataObject{T1,V1,V2}, sys::IdDSisoRational{T2,S})
  n = sys.info.n
  return predict(data, n, get_param(sys,n), sys.info.method)
end
