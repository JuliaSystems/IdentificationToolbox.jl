immutable IdOptions{T}
    OptimizationOptions::T
    estimate_initial::Bool
    loss_function::DistanceLoss
end

function IdOptions{L<:DistanceLoss}(;
    estimate_initial::Bool=true, loss_function::L=L2DistLoss(),
    autodiff::Bool=true, iterations::Int=10, kwargs...)
  OptimizationOptions = Optim.Options(;autodiff=autodiff, iterations=iterations, kwargs...)
  IdOptions{typeof(OptimizationOptions)}(OptimizationOptions,
    estimate_initial,
    loss_function)
end
