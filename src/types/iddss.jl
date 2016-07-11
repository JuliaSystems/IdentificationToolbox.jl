type IdDSs{T<:Real, M1<:AbstractMatrix{T}, M2<:AbstractMatrix{T},
    M3<:AbstractMatrix{T}, M4<:AbstractMatrix{T}, M5<:AbstractMatrix{T},
    M6<:AbstractMatrix{T}, S<:IdInfo} <: ControlCore.MimoSystem
    A::M1
    B::M2
    C::M3
    D::M4
    K::M5
    nx::Int
    nu::Int
    ny::Int
    Sigma::M6
    Ts::Float64
    info::S

    @compat function (::Type{IdDSs}){M1,M2,M3,M4,M5,M6,S}(A::M1, B::M2,
      C::M3, D::M4, K::M5, Sigma::M6, Ts::Float64, info::S)

      na, ma  = size(A,1,2)
      nb, mb  = size(B,1,2)
      nc, mc  = size(C,1,2)

        T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D), eltype(K), eltype(Sigma))
        return new{T,M1,M2,M3,M4,M5,M6,S}(A, B, C, D, K, na, nb, nc, Sigma, Ts, info)
    end
end

ControlCore.samplingtime(s::IdDSs) = s.Ts
ControlCore.isdiscrete(s::IdDSs)   = true

summary(s::IdDSs) = string("ss(nx=", s.nx, ",nu=", s.nu, ",ny=", s.ny, ",Ts=",
  s.Ts, ")")

showcompact(io::IO, s::IdDSs) = print(io, summary(s))

function show(io::IO, s::IdDSs)
  println(io, "Discrete time state space model")
  println(io, "\tx[k+1] = Ax[k] + Bu[k]")
  println(io, "\ty[k]   = Cx[k] + Du[k]")
  print(io, "with nx=", s.nx, ", nu=", s.nu, ", ny=", s.ny, ", Ts=", s.Ts, ".")
end

function showall(io::IO, s::IdDSs)
  show(io, s)
  println(io, "System matrix (A):")
  println(io, s.A)
  println(io, "Input matrix (B):")
  println(io, s.B)
  println(io, "Output matrix (C):")
  println(io, s.C)
  println(io, "Feedforward matrix (D):")
  print(io, s.D)
end
