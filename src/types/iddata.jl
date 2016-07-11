#####################################################################
#=                     IDDATA
            Datatype for System Identification

N    :   number of samples
nu   :   number of input channel
ny   :   number of output channel
Ts   :   Sampling time
y    :   N by ny matrix
u    :   N by nu matrix

Author : Lars Lindemann @2015

                                                                   =#
#####################################################################

immutable IdDataObject{T<:Real,V1<:AbstractVector{T},V2<:AbstractVector{T}}
    y::V1
    u::V2
    Ts::Float64
    N::Int
    nu::Int
    ny::Int

    @compat function (::Type{IdDataObject}){T}(y::Array{T}, u::Array{T}, Ts::Float64)
        N   = size(y, 1);
        ny  = size(y, 2);
        nu  = size(u, 2);

        # Validating amount of samples
        if size(y, 1) != size(u, 1)
            error("Input and output need to have the same amount of samples")
        end

        # Validate sampling time
        if Ts < 0
            error("Ts must be a real, positive number")
        end
        new{T,typeof(y),typeof(u)}(y, u, Ts, N, nu, ny)
    end
end

#####################################################################
##                      Constructor Functions                      ##
#####################################################################
@doc """`IdData = IdData(y, u, Ts=1, outputnames="", inputnames="")`

Creates an IdDataObject that can be used for System Identification. y and u should have the data arranged in columns.
Use for example sysIdentData = IdData(y1,[u1 u2],Ts,"Out",["In1" "In2"])""" ->
function iddata(y::Array, u::Array, Ts::Real=1.)
    nu = size(u,2)
    ny = size(y,2)

    y,u = promote(y,u)
    return IdDataObject(y, u, convert(Float64,Ts))
end

#####################################################################
##                          Misc. Functions                        ##
#####################################################################
## INDEXING ##
Base.ndims(::IdDataObject) = 2
Base.size(d::IdDataObject) = (d.ny,d.nu)
Base.size(d::IdDataObject,i) = i<=2 ? size(d)[i] : 1
Base.length(d::IdDataObject) = size(d.y, 1)

#####################################################################
##                         Math Operators                          ##
#####################################################################

## EQUALITY ##
function ==(d1::IdDataObject, d2::IdDataObject)
    fields = [:Ts, :u, :y]
    for field in fields
        if getfield(d1,field) != getfield(d2,field)
            return false
        end
    end
    return true
end

#####################################################################
##                        Display Functions                        ##
#####################################################################
Base.print(io::IO, d::IdDataObject) = show(io, d)

function Base.show(io::IO, dat::IdDataObject)
    println(io, "Discrete-time data set with $(dat.N) samples.")
    println(io, "Sampling time: $(dat.Ts) seconds")
end
function Base.showall(io::IO, dat::IdDataObject)
    println(io, "Discrete-time data set with $(dat.N) samples.")
    println(io, "Sampling time: $(dat.Ts) seconds")
end
