module IdentificationToolbox

import Base.==

# Import printing functions
import Base: showcompact, show, showall, summary

import ControlCore: isdiscrete, samplingtime, numstates, numinputs, numoutputs
import ControlCore: Siso, Continuous

export
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
  mse,
  modelfit,
  cost,
  predict,
  fitmodel

using Polynomials, Optim, PolynomialMatrices, ControlCore, ToeplitzMatrices,
  GeneralizedSchurAlgorithm, Compat

typealias PolyMatrix PolynomialMatrices.PolyMatrix
typealias Poly Polynomials.Poly

# types
include("types/iddata.jl")
include("types/idmodels.jl")
include("types/idmethods.jl")
include("types/idinfo.jl")
include("types/idmfd.jl")
include("types/iddss.jl")


# utilities
include("utilities/filt.jl")
include("utilities/filtic.jl")
include("utilities/detrend.jl")
include("utilities/compare.jl")

# methods
include("methods/pem.jl")
include("methods/armax.jl")
#include("methods/bj.jl")
include("methods/oe.jl")
#include("methods/arx.jl")
#include("methods/fir.jl")
#include("methods/stmcb.jl")
#include("methods/morsm.jl")
#include("methods/n4sid.jl")

end # module
