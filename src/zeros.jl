import Base: +, -, *,/, reshape, broadcasted

"""
    Zeros()

Acts as a stand-in for an array of zeros that can be
used during training which is ignored by the optimisers.

Useful to turn bias off for a forward pass of a layer.

## Examples

```julia-repl
julia> bias_less_conv = Conv((2,2), 1=>3; bias = false)
Conv((2, 2), 1=>3)

julia> params(bias_less_conv) |> length
1

julia> bias_less_conv.bias
Flux.Zeros()
```
"""
struct Zeros{T,N} <: AbstractArray{T,N}
  dims::NTuple{N,Int}
end

Zeros(::Type{T}, dims...) where T = Zeros{T,length(dims)}(dims)
Zeros(dims...) = Zeros(Bool, dims...)

Base.reshape(x::Zeros{T}, dims::Union{Colon,Int}...) where T = Zeros(T, Base._reshape_uncolon(x, dims)...)
Base.getindex(z::Zeros, args...) = error("Calling getindex on Zeros object, materialize to normal array or check for correctness")
Base.collect(x::Zeros{T}) where T = zeros(T, x.dims...)

Base.size(xs::Zeros) = xs.dims
Base.copyto!(a::Zeros, b::Zeros) = b

Base.collect(xs::Zeros{T,N}) where {T,N} = fill(zero(T), size(xs))

@adjoint reshape(xs::Zeros{T}, dims...) where T =
                reshape(xs, dims...), _ -> nothing

# Define basic ops
for f in (:+, :-)
  @eval @inline function $f(a::Union{AbstractArray{<:Number}, Zeros}, b::Zeros)
    @assert size(a) == size(b) throw(DimensionMismatch("dimensions must match"))
    a
  end
end

+(a::Zeros, b::AbstractArray) = b + a
-(a::Zeros, b::AbstractArray) = -b + a

Base.copy(xs::Zeros{T,N}) where {T,N} = xs

for op in (:+, :-)
  @eval function broadcasted(::typeof($op), a::AbstractArray, b::Zeros)
    bs = Broadcast.broadcast_shape(size(a), size(b))
    size(a) == bs && return a
    sz = similar(a, bs)
    sz .= a
  end
end

broadcasted(::typeof(+), a::Zeros, b::AbstractArray) = broadcasted(+, b, a)
broadcasted(::typeof(-), a::Zeros, b::AbstractArray) = broadcasted(+, -b, a)

for op in (:+, :-)
  @eval @adjoint function Base.broadcasted(::typeof($op), a::AbstractArray{T,N}, b::Zeros{S,M}) where {T <: Number, S <: Number, N,M}
    Base.broadcasted($op, a, b), Δ -> begin
      dims = M > N ? tuple(setdiff(1:M, 1:N)...) : tuple(setdiff(1:N, 1:M)...)
      da = dims == tuple(1:N...) ? Δ : dropdims(sum(Δ, dims = dims), dims = dims)
      (nothing, transpose(da), nothing)
    end
  end
end

Base.sum(z::Zeros{T}) where T = zero(T)

for op in (:+, :-)
  @eval @adjoint function $op(a::AbstractArray{T,N}, b::Zeros{S,M}) where {T <: Number, S <: Number, N,M}
    $op(a, b), Δ -> begin
      (Δ, nothing)
    end
  end
end

# Some opportunities to avoid scalar indexing, intermediaries
# Since it replicates a little of what we expect Base to do,
# it should be possible to remove in the future, but for now,
# these help with performance.
broadcasted(::typeof(+), a::AbstractArray, b::Zeros{T,0}) where T = a
broadcasted(::typeof(+), a::Zeros{T,0}, b::AbstractArray) where T = b
broadcasted(::typeof(-), a::AbstractArray, b::Zeros{T,0}) where T = a
broadcasted(::typeof(-), a::Zeros{T,0}, b::AbstractArray) where T = -b
