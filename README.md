# IdentificationToolbox

System identification toolbox for Julia.

### Build Status and Code Coverage

-  Build status: [![Build Status][build-img]][build-link]
-  Code coverage: [![Coveralls][ca-img]][ca-link] [![Codecov][cc-img]][cc-link]

[build-img]:  https://travis-ci.org/KTH-AC/IdentificationToolbox.jl.jl.svg?branch=master
[build-link]: https://travis-ci.org/KTH-AC/IdentificationToolbox.jl.jl
[ca-img]: https://coveralls.io/repos/github/KTH-AC/IdentificationToolbox.jl.jl/badge.svg?branch=master
[ca-link]: https://coveralls.io/github/KTH-AC/IdentificationToolbox.jl.jl?branch=master
[cc-img]: https://codecov.io/gh/KTH-AC/IdentificationToolbox.jl.jl/branch/master/graph/badge.svg
[cc-link]: https://codecov.io/gh/KTH-AC/IdentificationToolbox.jl.jl

### Description

This package is meant to provided methods for constructing mathematical models
of dynamic systems based on input-output data.

```julia
using IdentificationToolbox
u = randn(1000)
y = filt([0,1],[1,0.7],u) + randn(1000)
data = iddata(y,u,1)
model = pem(data, [2,2,1], ARX())
```
