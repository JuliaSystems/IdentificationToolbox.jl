@doc """`d0 = detrend(d)"")`
Returns a copy of IdData ´d´ where the mean has been subtracted from each signal.
""" ->
function detrend(d::IdDataObject)
    return IdDataObject(d.y.-mean(d.y,1), d.u.-mean(d.u,1), d.Ts, d.outputnames, d.inputnames)
end

@doc """`detrend!(d)"")`
Subtracts the mean from each signal in IdData ´d´.
""" ->
function detrend!(d::IdDataObject)
    d.y = d.y.-mean(d.y,1)
    d.u = d.u.-mean(d.u,1)
    return d
end
