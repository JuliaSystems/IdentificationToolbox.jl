module IdentificationToolbox

using Polynomials, Optim, PolynomialMatrices, LTISystems, ToeplitzMatrices,
  Compat, GeneralizedSchurAlgorithm

import Base: ==, size, length, getindex

# Import printing functions
import Base: showcompact, show, showall, summary
import ControlToolbox.dare

import LTISystems: isdiscrete, samplingtime, numstates, numinputs, numoutputs
import LTISystems: ss, tf, lfd

# using some PolynomialMatrices filtering
using PolynomialMatrices._filt_fir!
using PolynomialMatrices._filt_ar!

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
  fitmodel,
  iddata,
  IdOptions

using Reexport
@reexport using LossFunctions

const PolyMatrix = PolynomialMatrices.PolyMatrix
const Poly = Polynomials.Poly

# types
include("types/iddata.jl")
include("types/idmodels.jl")
include("types/idmethods.jl")
include("types/idinfo.jl")
include("types/idmfd.jl")
include("types/idstatespace.jl")
include("types/options.jl")

# utilities
#include("utilities/filt.jl")
include("utilities/filtic.jl")
include("utilities/detrend.jl")
include("utilities/compare.jl")

# methods
include("methods/pem.jl")
include("methods/siso_poly_pem.jl")
include("methods/arx.jl")
include("methods/stmcb.jl")
include("methods/morsm.jl")
include("methods/n4sid.jl")

end # module
