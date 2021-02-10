module PhyInfoNN
using Zygote
export @D

macro D(f)
    return :((x, θ) -> Zygote.pullback(y -> $f(y, θ), x)[2](ones(1, size(x, 2)))[1] )
end

end
