module Optimise

export train!,
  Descent, ADAM, Momentum, Nesterov, RMSProp, update!

include("optimisers.jl")
include("train.jl")

end