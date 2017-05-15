abstract IdInfo{S}

immutable IterativeIdInfo{S} <: IdInfo{S} #{T<:IterativeIdMethod}
  mse::Vector{Float64}
  modelfit::Vector{Float64}
  opt::Optim.OptimizationResults
  model::IdModel{S}
end

immutable OneStepIdInfo{S} <: IdInfo{S} #{T<:OneStepIdMethod}
  mse::Vector{Float64}
  modelfit::Vector{Float64}
  model::IdModel{S}
end
