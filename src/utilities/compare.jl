@doc """`MSE, fit = compare(m, d)`
Compare model `m` with validation data `d` using one step ahead prediction. Returns the mean square error `MSE` and the fit value `fit = 100 * (1 - √MSE / (y-mean(y)))`. See also `pred`.
""" ->
function compare(m::IdDSisoRational, d::IdDataObject)
    # compare true system with estimated model (validation)
    y = d.y
    y_est = predict(m, d)
    M = timehorizon(m)

    E = sum(x->x^2, y-y_est)
    fit = 100 * (1 - sqrt(E)/norm(y[M:end]-mean(y[M:end])))
    return E/(length(d)-M), fit
end
