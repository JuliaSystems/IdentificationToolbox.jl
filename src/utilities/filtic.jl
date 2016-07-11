"""
  filtic(b,a,y[,x]) -> si

Returns the initial state of the filter defined by (`b`,`a`)

The vectors `y` and `x` are stored with more recent values first.

filtic(b,a,y) assumes past avlues of `x` are zero.

# Examples
```julia
julia> u = [1;0;0;0];

julia> a = [1;2];

julia> b = [3;4];

julia> y = filt(b,a,u)
4-element Array{Float64,1}:
  3
 -2
  4
 -8

julia> si = filtic(b,a,y[1:1],u[1:1])
1-element Array{Float64,1}:
 -2.0

julia> filt(b,a,u[2:end],si)
3-element Array{Float64,1}:
 -2
  4
 -8
```
"""
function filtic{T,S}(b::Union{AbstractVector, Number}, a::Union{AbstractVector, Number},
    y::AbstractArray{T}, x::AbstractArray{S}=zeros(length(b)-1))
  na = length(a)
  nb = length(b)
  m = max(na,nb)-1

  # Pad the coefficients with zeros if needed
  length(x) < nb-1   && (x = copy!(zeros(eltype(x), nb-1), x))
  length(y) < na-1   && (y = copy!(zeros(eltype(y), na-1), y))

  # Filter coefficient normalization
  if a[1] != 1
    norml = a[1]
    a ./= norml
    b ./= norml
  end

  si = zeros(m)
  if na-1 > 0
    si[na-1:-1:1] += filt(-a[na:-1:2], 1, y[1:na-1])
  end
  if nb-1 > 0
    si[nb-1:-1:1] += filt(b[nb:-1:2], 1, x[1:nb-1])
  end
  si
end
