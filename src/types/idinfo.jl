abstract IdInfo

immutable IterativeIdInfo{T<:IterativeIdMethod} <: IdInfo
  mse::Float64
  modelfit::Float64
  opt::Optim.OptimizationResults
  method::T
  n::Vector{Int}
end

immutable OneStepIdInfo{T<:OneStepIdMethod} <: IdInfo
  mse::Float64
  modelfit::Float64
  method::T
  n::Vector{Int}
end
