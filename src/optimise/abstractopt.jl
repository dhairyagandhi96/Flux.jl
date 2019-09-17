using Flux
using Base: @get!
using MacroTools: @forward

abstract type AbstractOptimiser end

"""
    ADAM(η = 0.001, β = (0.9, 0.999))

[ADAM](https://arxiv.org/abs/1412.6980v8) optimiser.
"""
struct ADAM{K,T} <: AbstractOptimiser
  eta::K
  beta::T
end

# ADAM(η = 0.001, β = (0.9, 0.999)) = ADAM(η, β)
ADAM() = ADAM(0.001, (0.9, 0.999))


function apply!(o::ADAM, x, Δ, state)
  η, β = o.eta, o.beta
  mt, vt, βp = get!(state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  s = (mt, vt, βp .* β)
  return Δ, s
end


