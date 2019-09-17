using Juno
import Zygote: Params, gradient
import Zygote.IRTools: @dynamo, recurse!, IR

function _apply!(opt, x, x̄, state::IdDict)
  Δ, s = apply!(opt, x, x̄, state)
  st = copy(state)
  st[x] = s
  Δ, st
end

function _update!(opt, x, x̄, state)
  Δ, state = _apply!(opt, x, x̄, state)
  x .- Δ, state
end

update!(opt, xs::Params, gs, state = IdDict()) = _update!(opt, xs, gs, state)

function _update!(opt, xs::Params, gs, state::IdDict)
  ps = copy(xs.order)
  for (i,x) in enumerate(xs)
    if gs[x] isa Nothing
       ps[i] = x
       continue
    end
    d, state = _update!(opt, x, gs[x], state)
    # x .= d
    ps[i] = d
  end

  Params(ps), state
end

if VERSION >= v"1.3"
  apply!(opt::AbstractOptimiser, ps, gs, state) = error("no `apply!` definition found")
end

"""
  `apply!(opt, ps, gs, state)`

  Define the update rules for an optimsier `opt`, given the params, `ps`, gradients `gs` and the current state.
"""
function apply!(opt, x, Δ, state) end
