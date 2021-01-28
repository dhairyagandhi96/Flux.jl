using Tracker
using .Optimise

tracker(m) = fmap(Tracker.param, m)
data(m) = fmap(Tracker.data, m)

function Flux.Optimise.update!(opt, xs::Tracker.Params, gs)
  for x in xs
    Optimise.update!(opt, x, gs[x])
  end
end

function Optimise.update!(opt, x::TrackedArray, x̄)
  Tracker.update!(x, -Optimise.apply!(opt, Tracker.data(x), Tracker.data(x̄)))
end
