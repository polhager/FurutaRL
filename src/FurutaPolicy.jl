export FurutaPolicy, action

using ReinforcementLearning
using LinearAlgebra

struct FurutaPolicy{S} <: AbstractPolicy
    action_space::S
end

#Classical controller for the Furuta pendulum.

function FurutaPolicy(;s=nothing)
    FurutaPolicy(s)
end

function (p::FurutaPolicy)(env::AbstractEnv)
    x = state(env)

    sθ, cθ, θ̇ , ϕ, ϕ̇  = x
    θ = angle_norm(env.env.state[1])

    if abs(θ) < 0.5 && abs(θ̇ ) < 5
        L = 1/env.env.params.max_torque .* [-0.8139, -0.0541, -0.0226, -0.0258]

        u = -dot(L, [θ, θ̇ , ϕ, ϕ̇ ])

        if abs(ϕ̇ ) > 0.1
            uf = 0.0076*sign(ϕ̇ )/env.env.params.max_torque
        else
            uf = 0.008*sign(u)/env.env.params.max_torque
        end
        u += uf
    elseif abs(angle_norm(θ-π)) < 0.1 && abs(θ̇ ) < 1
        u = 1.0
    else
        En = (cθ - 1 + θ̇ ^2/(2*15.4520^2))
        u = θ̇ *cθ*En/(10 + abs(θ̇ *cθ*En))
    end

    u = clamp(u, -1.0, 1.0)
    return u
end

function action(p::FurutaPolicy, s; training = false)
    sθ, cθ, θ̇ , ϕ, ϕ̇  = s
    θ = angle_norm(atan(sθ, cθ))
    # θ, θ̇ , ϕ, ϕ̇  = s

    if abs(θ) < 0.5 && abs(θ̇ ) < 5
        L = 1/0.04 .* [-0.8139, -0.0541, -0.0226, -0.0258]

        u = -dot(L, [θ, θ̇ , ϕ, ϕ̇ ])

        if abs(ϕ̇ ) > 0.1
            uf = 0.0076*sign(ϕ̇ )/0.04
        else
            uf = 0.008*sign(u)/0.04
        end
        u += uf
    elseif abs(angle_norm(θ-π)) < 0.1 && abs(θ̇ ) < 1
        u = 1.0
    else
        En = (cθ - 1 + θ̇ ^2/(2*15.4520^2))
        u = θ̇ *cθ*En/(10 + abs(θ̇ *cθ*En))
    end

    u = clamp(u, -1.0, 1.0)
    return u
end

function (p::FurutaPolicy)(s)

    sθ, cθ, θ̇ , ϕ, ϕ̇  = s
    θ = atan(sθ, cθ)

    if abs(θ) < 0.5 && abs(θ̇ ) < 5
        L = 1/p.action_space.right .* [-0.8139, -0.0541, -0.0226, -0.0258]

        u = -dot(L, [θ, θ̇ , ϕ, ϕ̇ ]) + 0.0076*sign(ϕ̇ )/p.action_space.right

        if abs(ϕ̇ ) > 0.1
            uf = 0.0076*sign(ϕ̇ )/p.action_space.right
        else
            uf = 0.008*sign(u)/p.action_space.right
        end
        u += uf
    elseif abs(angle_norm(θ-π)) < 0.1 && abs(θ̇ ) < 1
        u = 1.0
    else
        En = (cθ - 1 + θ̇ ^2/(2*15.4520^2))
        u = θ̇ *cθ*En/(10 + abs(θ̇ *cθ*En))
    end

    u = clamp(u, -1.0, 1.0)
    return u
end

RLBase.update!(p::FurutaPolicy, x) = nothing
angle_norm(x) = Base.mod((x + Base.π), (2 * Base.π)) - Base.π
