# IdentificationToolbox

[![Unix][unix-img]][unix-link]
[![Windows][win-img]][win-link]
[![Coveralls][ca-img]][ca-link]
[![Codecov][cc-img]][cc-link]
[![Documentation][docs-latest-img]][docs-latest-link]
[![Gitter][gitter-img]][gitter-link]

[unix-img]: https://img.shields.io/travis/JuliaSystems/IdentificationToolbox.jl/master.svg?label=unix
[unix-link]: https://travis-ci.org/JuliaSystems/IdentificationToolbox.jl
[win-img]: https://img.shields.io/appveyor/ci/aytekinar/identificationtoolbox-jl/master.svg?label=windows
[win-link]: https://ci.appveyor.com/project/aytekinar/identificationtoolbox-jl/branch/master
[ca-img]: https://img.shields.io/coveralls/JuliaSystems/IdentificationToolbox.jl/master.svg?label=coveralls
[ca-link]: https://coveralls.io/github/JuliaSystems/IdentificationToolbox.jl?branch=master
[cc-img]: https://img.shields.io/codecov/c/github/JuliaSystems/IdentificationToolbox.jl/master.svg?label=codecov
[cc-link]: https://codecov.io/gh/JuliaSystems/IdentificationToolbox.jl?branch=master
[docs-latest-img]: https://img.shields.io/badge/documentation-latest-blue.svg?colorB=1954a6
[docs-latest-link]: https://identificationtoolbox.readthedocs.io/en/latest
[gitter-img]: https://img.shields.io/gitter/room/JuliaSystems/IdentificationToolbox.jl.svg?colorB=1954a6
[gitter-link]: https://gitter.im/JuliaSystems/IdentificationToolbox.jl

System identification toolbox for Julia.

### Description

This package is meant to provided methods for constructing mathematical models
of dynamic systems based on input-output data.

```julia
# Use `Pkg.clone` until the package is registered in `METADATA.jl`
Pkg.clone("https://github.com/JuliaSystems/IdentificationToolbox.jl")
using IdentificationToolbox
u = randn(1000)
y = filt([0,1],[1,0.7],u) + randn(1000)
data = iddata(y,u,1)
model = pem(data, [2,2,1], ARX())
```
