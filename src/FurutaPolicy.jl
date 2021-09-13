export FurutaPolicy, action

using ReinforcementLearning
using LinearAlgebra

struct FurutaPolicy{S} <: AbstractPolicy
    action_space::S
end

function FurutaPolicy(;s=nothing)
    FurutaPolicy(s)
end

# function (p::FurutaPolicy)(env)
#     sθ, cθ, θ̇ , ϕ, ϕ̇  = state(env)
#     θ = atan(sθ, cθ)

#     if abs(θ) < 0.1 && abs(θ̇ ) < 5
#         L = [-2.3312, -0.1552, -0.0688, -0.0782]
#         u = -dot(L, [θ, θ̇ , ϕ, ϕ̇ ]) + 0.0076*sign(ϕ̇ )
#     elseif abs(θ-π) < 0.01 && abs(θ̇ ) < 1
#         u = s.right
#     else
#         En = (cos(θ) - 1 + θ̇ ^2/(2*15.4520^2))
#         u = θ̇ *cos(θ)*En/(30 + abs(θ̇ *cos(θ)*En))
#     end

#     u = clamp(u, action_space(env).left, action_space(env).right)
#     return u
# end

function (p::FurutaPolicy)(env::AbstractEnv)
    x = state(env)

    sθ, cθ, θ̇ , ϕ, ϕ̇  = x
    #θ = atan(sθ, cθ)
    θ = angle_norm(env.env.state[1])

    if abs(θ) < 0.5 && abs(θ̇ ) < 5
        L = 1/env.env.params.max_torque .* [-0.8139, -0.0541, -0.0226, -0.0258]
        #L = 1/env.env.params.max_torque .* [-0.4914, -0.0326, -0.0128, -0.0146]

        u = -dot(L, [θ, θ̇ , ϕ, ϕ̇ ]) #+ 0.0076*sign(ϕ̇ )/env.env.params.max_torque

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
        #u = θ̇ *cos(θ)*En/(60 + abs(θ̇ *cos(θ)*En))/env.env.params.max_torque
        #u = .000075 * (cθ * cθ * cθ * cθ) * θ̇ * (9.81 * (1.0 - cθ) - θ̇ * θ̇ * 0.0075)/0.04
        #u = 0.7*sign(θ̇ *cos(θ)*En)
    end

    u = clamp(u, -1.0, 1.0)
    return u
end

function action(p::FurutaPolicy, s; training = false)
    sθ, cθ, θ̇ , ϕ, ϕ̇  = s
    #θ = atan(sθ, cθ)
    θ = angle_norm(atan(sθ, cθ))
    # θ, θ̇ , ϕ, ϕ̇  = s
    # cθ = cos(θ)

    if abs(θ) < 0.5 && abs(θ̇ ) < 5
        L = 1/0.04 .* [-0.8139, -0.0541, -0.0226, -0.0258]
        #L = 1/env.env.params.max_torque .* [-0.4914, -0.0326, -0.0128, -0.0146]

        u = -dot(L, [θ, θ̇ , ϕ, ϕ̇ ]) #+ 0.0076*sign(ϕ̇ )/0.04

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
        #u = θ̇ *cos(θ)*En/(60 + abs(θ̇ *cos(θ)*En))/env.env.params.max_torque
        #u = .000075 * (cθ * cθ * cθ * cθ) * θ̇ * (9.81 * (1.0 - cθ) - θ̇ * θ̇ * 0.0075)/0.04
        #u = 0.7*sign(θ̇ *cos(θ)*En)
    end

    u = clamp(u, -1.0, 1.0)
    return u
end

function (p::FurutaPolicy)(s)

    sθ, cθ, θ̇ , ϕ, ϕ̇  = s
    θ = atan(sθ, cθ)

    if abs(θ) < 0.5 && abs(θ̇ ) < 5
        L = 1/p.action_space.right .* [-0.8139, -0.0541, -0.0226, -0.0258]
        #L = 1/env.env.params.max_torque .* [-0.4914, -0.0326, -0.0128, -0.0146]

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
        #u = θ̇ *cos(θ)*En/(60 + abs(θ̇ *cos(θ)*En))/env.env.params.max_torque
        #u = .000075 * (cθ * cθ * cθ * cθ) * θ̇ * (9.81 * (1.0 - cθ) - θ̇ * θ̇ * 0.0075)/0.04
        #u = 0.7*sign(θ̇ *cos(θ)*En)
    end

    u = clamp(u, -1.0, 1.0)
    return u
end

RLBase.update!(p::FurutaPolicy, x) = nothing
angle_norm(x) = Base.mod((x + Base.π), (2 * Base.π)) - Base.π
