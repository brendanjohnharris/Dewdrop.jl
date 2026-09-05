using CUDA, Adapt

struct Het{T}
    params::T
end
Adapt.@adapt_structure Het
Base.Broadcast.broadcastable(h::Het) = Ref(h)

_resting_of(h, i) = h.params[i]

V = CUDA.zeros(Float32, 5)
h = Het(CUDA.rand(Float32, 5))

# Try the broadcast
V .= Float32.(_resting_of.(h, 1:size(V, 1)))

println("Success! V = ", Array(V))
