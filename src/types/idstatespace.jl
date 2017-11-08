immutable IdStateSpace{T,S,M,I} <: LTISystems.LtiSystem{T,S}
  A::M
  B::M
  C::M
  D::M
  K::M
  nx::Int
  nu::Int
  ny::Int
  Sigma::M
  Ts::Float64
  info::I

  @compat function (::Type{IdStateSpace}){M,I}(A::M, B::M,
    C::M, D::M, K::M, Sigma::M, Ts::Float64, info::I)
    na  = size(A,1)
    nb  = size(B,2)
    nc  = size(C,1)
    return new{Val{:mimo},Val{:disc},M,I}(A, B, C, D, K, na, nb, nc, Sigma, Ts, info)
  end

  @compat function (::Type{IdStateSpace}){M,I}(A::M, B::M,
    C::M, d::Real, K::M, Sigma::M, Ts::Float64, info::I)
    na  = size(A,1)
    nb  = size(B,2)
    nc  = size(C,1)
    return new{Val{:siso},Val{:disc},M,I}(A, B, C, fill(d,1,1), K, na, nb, nc, Sigma, Ts, info)
  end
end

# conversion to state-space
ss(s::IdStateSpace{Val{:siso},Val{:disc}}) = ss(s.A, s.B, s.C, s.D[1], s.Ts)
ss(s::IdStateSpace{Val{:mimo},Val{:disc}}) = ss(s.A, s.B, s.C, s.D, s.Ts)

samplingtime(s::IdStateSpace) = s.Ts
isdiscrete(s::IdStateSpace)   = true

numstates(s::IdStateSpace)    = s.nx
numinputs(s::IdStateSpace)    = s.nu
numoutputs(s::IdStateSpace)   = s.ny
