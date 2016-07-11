module IdentificationToolbox

import Base.==

# Import printing functions
import Base: showcompact, show, showall, summary

export
  # datatypes
  IdDataObject,
  IdDSs,
  IdDSisoRational,
  iddata,
  # utilities
  detrend,
  detrend!,
  dare,
  compare,
  filtic,
  # identification methods
  fir,    FIR,
  arx,    ARX,
  armax,  ARMAX,
  oe,     OE,
  bj,     BJ,
  n4sid,
  stmcb,  STMCB,
  morsm,  MORSM,
  pem,
  # method functions
  predict,
  fitmodel

using Polynomials, Optim, ControlCore, ToeplitzMatrices,
  GeneralizedSchurAlgorithm, Compat

# types
include("types/iddata.jl")
include("types/idmethods.jl")
include("types/idinfo.jl")
include("types/iddsisorational.jl")
include("types/iddss.jl")

# utilities
include("utilities/filtic.jl")
include("utilities/detrend.jl")
include("utilities/compare.jl")

# methods
include("methods/pem.jl")
include("methods/armax.jl")
include("methods/bj.jl")
include("methods/oe.jl")
include("methods/arx.jl")
include("methods/fir.jl")
include("methods/stmcb.jl")
include("methods/morsm.jl")
include("methods/n4sid.jl")

end # module
