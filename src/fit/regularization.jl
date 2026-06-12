#=
regularization.jl

    Provides a collection of types for regularization components used in
    estimation for dynamic factor models.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/12/27
=#

"""
    NormL21Weighted(λ, dim=1)

Return the "sum of ``ℓ₂`` norm" function

```math
f(X) = ∑ λᵢ ⋅ ||xᵢ||
```

for a nonnegative `λ` array, where ``xᵢ`` is the ``i``-th column of ``X`` if `dim == 1`, and
the ``i``-th row of ``X`` if `dim == 2`. In words, it is the sum of the Euclidean norms of
the columns or rows.
"""
struct NormL21Weighted{V <: AbstractVector, I}
    λ::V
    dim::I
    function NormL21Weighted{V, I}(λ::V, dim::I) where {V, I}
        if any(λ .< 0)
            error("parameter λ must be nonnegative")
        else
            new(λ, dim)
        end
    end
end

NormL21Weighted(λ::V, dim::I = 1) where {V, I} = NormL21Weighted{V, I}(λ, dim)

function (f::NormL21Weighted)(X)
    R = real(eltype(X))
    nslice = R(0)
    n21X = R(0)
    if f.dim == 1
        for j in axes(X, 2)
            nslice = R(0)
            for i in axes(X, 1)
                nslice += abs(X[i, j])^2
            end
            n21X += f.λ[j] * sqrt(nslice)
        end
    elseif f.dim == 2
        for i in axes(X, 1)
            nslice = R(0)
            for j in axes(X, 2)
                nslice += abs(X[i, j])^2
            end
            n21X += f.λ[i] * sqrt(nslice)
        end
    end

    return n21X
end

function prox!(Y, f::NormL21Weighted, X, γ)
    R = real(eltype(X))
    nslice = R(0)
    n21X = R(0)
    if f.dim == 1
        for j in axes(X, 2)
            gl = γ * f.λ[j]
            nslice = R(0)
            for i in axes(X, 1)
                nslice += abs(X[i, j])^2
            end
            nslice = sqrt(nslice)
            scal = 1 - gl / nslice
            scal = scal <= 0 ? R(0) : scal
            for i in axes(X, 1)
                Y[i, j] = scal * X[i, j]
            end
            n21X += f.λ[j] * scal * nslice
        end
    elseif f.dim == 2
        for i in axes(X, 1)
            gl = γ * f.λ[i]
            nslice = R(0)
            for j in axes(X, 2)
                nslice += abs(X[i, j])^2
            end
            nslice = sqrt(nslice)
            scal = 1 - gl / nslice
            scal = scal <= 0 ? R(0) : scal
            for j in axes(X, 2)
                Y[i, j] = scal * X[i, j]
            end
            n21X += f.λ[i] * scal * nslice
        end
    end

    return n21X
end

"""
    NormL1plusL21(λ, γ, dim)

With two nonegative scalars λ and γ, return the function

```math
f(X) = λ ⋅ ∑ |xᵢⱼ| + γ ⋅ ∑ ||xₖ||
```

and with a nonnegative array parameters λ and γ, return the function

```math
f(X) = ∑ λᵢⱼ ⋅ |xᵢⱼ| + ∑ γₖ ⋅ ||xₖ||
```

where ``xₖ`` is the ``k``-th column of ``X`` if `dim == 1`, and the ``k``-th row of ``X`` if
`dim == 2`. In words, it is the sum of the ``ℓ₁``-norm and sum of the Euclidean norms of the
columns or rows.
"""
struct NormL1plusL21{L1 <: NormL1, L21 <: Union{NormL21, NormL21Weighted}}
    l1::L1
    l21::L21
end

function NormL1plusL21(λ::Real = 1, γ::Real = 1, dim::Int = 1)
    NormL1plusL21(NormL1(λ), NormL21(γ, dim))
end
function NormL1plusL21(λ::T, γ::AbstractVector, dim::Int = 1) where {T}
    NormL1plusL21(NormL1(λ), NormL21Weighted(γ, dim))
end

(f::NormL1plusL21)(x) = f.l1(x) + f.l21(x)

function prox!(y, f::NormL1plusL21, x, γ)
    prox!(y, f.l1, x, γ)
    fl21 = prox!(y, f.l21, y, γ)

    return f.l1(y) + fl21
end

"""
    TotalVariation1DWeighted(λ)

Return the 1D total variation function

```math
f(x) = ∑ λᵢ ⋅ |xᵢ₊₁ - xᵢ|
``` 

for a nonnegative `λ` array. In words, it is the sum of the absolute differences between
adjacent elements of `x`, weighted by `λ`.
"""
struct TotalVariation1DWeighted{V <: AbstractVector}
    λ::V
    function TotalVariation1DWeighted{V}(λ::V) where {V}
        if any(λ .< 0)
            error("parameter λ must be nonnegative")
        else
            new(λ)
        end
    end
end

TotalVariation1DWeighted(λ::V) where {V} = TotalVariation1DWeighted{V}(λ)

(f::TotalVariation1DWeighted)(x) = sum(f.λ[i] * abs(x[i + 1] - x[i]) for i in eachindex(f.λ))

function prox!(y, f::TotalVariation1DWeighted, x, γ)
    # solves y = arg min_z sum_{k} lam[k] |z_{k+1}-z_k| + 1/2 * ||z-x||^2
    N = length(x)
    if N == 0 return end
    if N == 1
        y[1] = x[1]
        return
    end

    # Pad the penalties with a trailing zero to eliminate the k==N edge case
    L = zeros(eltype(x), N)
    L[1:N-1] .= γ .* f.λ

    k0 = kminus = kplus = 1
    vmin = x[1] - L[1]
    vmax = x[1] + L[1]
    umin = L[1]
    umax = -L[1]

    k = 2
    while k <= N
        umin += x[k] - vmin
        umax += x[k] - vmax
        
        # 1. Negative jump
        if umin < -L[k]
            y[k0:kminus] .= vmin
            k0 = kminus + 1
            
            if k0 > N; break; end
            
            k = kminus = kplus = k0
            vmin = x[k0] + L[k0-1] - L[k0]
            vmax = x[k0] + L[k0-1] + L[k0]
            umin = L[k0]
            umax = -L[k0]
            k += 1
            
        # 2. Positive jump
        elseif umax > L[k]
            y[k0:kplus] .= vmax
            k0 = kplus + 1
            
            if k0 > N; break; end
            
            k = kminus = kplus = k0
            vmin = x[k0] - L[k0-1] - L[k0]
            vmax = x[k0] - L[k0-1] + L[k0]
            umin = L[k0]
            umax = -L[k0]
            k += 1
            
        # 3. Update candidate slopes
        else
            if umin >= L[k]
                vmin += (umin - L[k]) / (k - k0 + 1)
                umin = L[k]
                kminus = k
            end
            if umax <= -L[k]
                vmax += (umax + L[k]) / (k - k0 + 1)
                umax = -L[k]
                kplus = k
            end
            k += 1
        end
    end

    # Safely fill the remaining tail segment
    if k0 <= N
        y[k0:N] .= vmin
    end

    return f(y)
end