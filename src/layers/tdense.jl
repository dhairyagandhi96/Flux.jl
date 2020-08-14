struct TDense{F,T,S}
    σ::F
    W::T
    b::S
end
@functor TDense
TDense(W::AbstractArray, b::AbstractArray, σ = identity) =
  TDense(σ, W, b)
function TDense(in::Int, out::Int, σ = identity)
  σ = typeof(σ) <: typeof(relu) ? vrelu : σ
  TDense(σ, rand(Float32, out, in), zeros(Float32, out))
end
function denselayer!(res, W, X, b)
   @avx for o ∈ axes(W,1), n ∈ axes(res,2)
       res2 = zero(eltype(res))#b[o]
       for i ∈ axes(W,2)
           res2 += W[o,i] * X[i,n]
       end
       res[o,n] = res2 + b[o]
   end
end
function matmul!(res, W, X)
   @avx for o ∈ axes(W,1), n ∈ axes(res,2)
       res2 = zero(eltype(res))#b[o]
       for i ∈ axes(W,2)
           res2 += W[o,i] * X[i,n]
       end
       res[o,n] = res2
   end
end
function matmul_threaded!(res, W, X)
   N = size(X,2); nthreads = min(Base.Threads.nthreads(), N >> 7)
   nthreads > 1 || return matmul!(res, W, X)
   Base.Threads.@sync for t ∈ 1:nthreads
       Base.Threads.@spawn begin
           nstart = 1 + (((t - 1)*N) ÷ nthreads)
           nstop = (t*N) ÷ nthreads
           matmul!(view(res, :, nstart:nstop), W, view(X, :, nstart:nstop))
       end
   end
end
function denselayer_threaded!(res, dense, X)
   N = size(X,2); nthreads = min(Base.Threads.nthreads(), N >> 7)
   nthreads > 1 || return denselayer!(res, dense.W, X, dense.b)
   Base.Threads.@sync for t ∈ 1:nthreads
       Base.Threads.@spawn begin
           nstart = 1 + (((t - 1)*N) ÷ nthreads)
           nstop = (t*N) ÷ nthreads
           denselayer!(view(res, :, nstart:nstop), dense.W, view(X, :, nstart:nstop), dense.b)
       end
   end
end
function (a::TDense)(x)
    res = similar(x, size(a.W, 1), size(x, 2))
    denselayer_threaded!(res, a, x)
    @avx vmap(a.σ, res)
#     res
end
vrelu(x::T, z = zero(T)) where T = vifelse(x > z, x, z)
vdrelu(x::T, Δ, z = zero(T)) where T = vifelse(x > z, Δ, z)
@adjoint function (a::TDense)(x)
  y = similar(x, size(a.W, 1), size(x, 2))
  denselayer_threaded!(y, a, x)
  function back(Δ)
    # Hopefully you're using tanh, or identity, although
    # we should also add relu and a couple other common ones
    z̄ = if typeof(a.σ) <: typeof(tanh)
          @avx Δ .* (1 .- a.σ.(y) .* a.σ.(y))
        elseif typeof(a.σ) <: typeof(relu) || typeof(a.σ) <: typeof(vrelu)
          @avx vmap(vdrelu, x, Δ)
        elseif typeof(a.σ) <: typeof(identity)
          Δ .* y
        else
          error("I haven't added the gradient for other activations yet.
                 Or rather, just use `pullback` here and call it a day.
                 Add gradient for $(a.σ)")
        end
    W̄ = similar(a.W)
    x̄ = similar(x)
    matmul_threaded!(W̄, z̄, x')
    b̄ = z̄
    matmul_threaded!(x̄, a.W', z̄)
    ((σ = nothing, W = W̄, b = b̄), x̄)
  end
  @avx a.σ.(y), back
end
