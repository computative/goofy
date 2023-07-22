#julia> typeof(raw)
#Dict{String, Any}
#
#julia> typeof(raw["0"])
#Vector{Any} (alias for Array{Any, 1})
#
#julia> typeof(raw["0"][1])
#Vector{Any} (alias for Array{Any, 1})
#
#julia> typeof(raw["1"][1][1])

using JSON

raw = JSON.parsefile("/home/marius/Dokumenter/Skole/phd/goofy.git/test/1.json")

m = length(raw)
n = length(raw["1"])
l = length(raw["1"][1])

v::Array{Float64,3} = Array{Float64}(undef, m, n, l)

for (i, (key, value)) in enumerate(raw)
    v[:, :, i] .= hcat([cat(Float64, vec...) for vec in value]...)
end

println( typeof(v) )