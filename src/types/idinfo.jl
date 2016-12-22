abstract IdInfo

immutable IterativeIdInfo <: IdInfo #{T<:IterativeIdMethod}
  mse::Vector{Float64}
  modelfit::Vector{Float64}
  opt::Optim.OptimizationResults
  model::IdModel
end

immutable OneStepIdInfo <: IdInfo #{T<:OneStepIdMethod}
  mse::Vector{Float64}
  modelfit::Vector{Float64}
  model::IdModel
end
