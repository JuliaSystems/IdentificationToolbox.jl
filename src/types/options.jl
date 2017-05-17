immutable IdOptions{T}
    OptimizationOptions::T
    estimate_initial::Bool
    loss_function::DistanceLoss
end

function IdOptions{L<:DistanceLoss}(;
    estimate_initial::Bool=false, store_trace::Bool=true, loss_function::L=L2DistLoss(),
    autodiff::Bool=false, iterations::Int=100, kwargs...)
  OptimizationOptions = Optim.Options(;autodiff=autodiff, iterations=iterations,store_trace=store_trace, kwargs...)
  IdOptions{typeof(OptimizationOptions)}(OptimizationOptions,
    estimate_initial,
    loss_function)
end
