export FurutaEnv, reset!, step!, actualState
using ReinforcementLearning
using Random
using IntervalSets

#Model of the Furuta pendulum, adapted from JuliaReinforcementLearning's pendulum model.

struct FurutaEnvParams{T}
    max_speed::T
    max_torque::T
    max_steps::Int
    dt::T
    J::T
    M::T
    ma::T
    mp::T
    la::T
    lp::T
    tau_C::T
    tau_S::T
end

mutable struct FurutaEnv{A,T,R<:AbstractRNG} <: AbstractEnv
    params::FurutaEnvParams{T}
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    done::Bool
    t::Int
    rng::R
    reward::T
end

function FurutaEnv(;
    T = Float32,
    max_speed = T(100),
    max_torque = T(0.04),
    max_steps = 5000,
    dt = T(0.002),
    J = T(1.54e-4),
    M = T(0),
    ma = T(0),
    mp = T(5.44e-3),
    la = T(4.3e-2),
    lp = T(6.46e-2),
    tau_C = T(0.0076),
    tau_S = T(0.008),
    rng = Random.GLOBAL_RNG,)
    high = T.([1, 1, max_speed, Inf64, max_speed])
    action_space = -max_torque..max_torque
    env = FurutaEnv(
        FurutaEnvParams(max_speed, max_torque, max_steps, dt, J,M,ma,mp,la,lp,tau_C,tau_S),
        action_space,
        Space(ClosedInterval{T}.(-high, high)),
        zeros(T, 4),
        false,
        0,
        rng,
        zero(T),
    )
    reset!(env)
    return env
end

Random.seed!(env::FurutaEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::FurutaEnv) = env.action_space
RLBase.state_space(env::FurutaEnv) = env.observation_space
RLBase.reward(env::FurutaEnv) = env.reward
RLBase.is_terminated(env::FurutaEnv) = env.done
RLBase.state(env::FurutaEnv) = _get_obs(env)
actualState(env::FurutaEnv) = env.state

function (env::FurutaEnv)(u)
    step!(env, u)
end

function step!(env::FurutaEnv, u)
    env.t += 1
    par = env.params
    J,M,ma,mp,la,lp = par.J,par.M,par.ma,par.mp,par.la,par.lp
    g = 9.81
    α = J+(M+ma/3+mp)*la^2
    β = (M+mp/3)*lp^2
    γ = (M+mp/2)*la*lp
    δ = (M+mp/2)*g*lp

    θ, θ̇, ϕ, ϕ̇ = env.state

    if abs(ϕ̇ ) > 0.1
        u = u - par.tau_C*sign(ϕ̇ )
    elseif abs(u) < par.tau_S
        u = 0
    else
        u = u - par.tau_S*sign(u)
    end
    u = clamp(u, -par.max_torque, par.max_torque)
    ψ = 1/(α*β - γ^2 + (β^2 + γ^2)*sin(θ)^2)

    r, done = _reward(env, u)
    steps = 1
    h = par.dt / steps
    y = env.state

    for i in 1:steps   
        #RK3/8
        k1 = _f(env, y, u)
        k2 = _f(env, y + h .* k1 ./ 3, u)
        k3 = _f(env, y + h .* (-k1 ./ 3 + k2), u)
        k4 = _f(env, y + h .* (k1 - k2 + k3), u)
        y = y .+ h .* (k1 ./ 8 + 3 .* k2 ./ 8 + 3 .*k3 ./ 8 + k4 ./ 8)
    end

    θ, θ̇, ϕ, ϕ̇ = y

    θ̇ = clamp(θ̇ , -par.max_speed, par.max_speed)
    ϕ̇ = clamp(ϕ̇ , -par.max_speed, par.max_speed)

    env.state = vcat(θ, θ̇ , ϕ, ϕ̇ )
    env.reward = r
    if env.t >= par.max_steps
        done = true
    end
    env.done = done

    return _get_obs(env), r, done, Dict()
end

function RLBase.reset!(env::FurutaEnv{A,T}) where {A,T}
    env.state = T.(vec([π 0 0 0]))
    if isdefined(Main, :CuArrays)
        env.state = env.state |> gpu
    end
    env.t = 0
    env.reward = T(0.0)
    env.done = false
    return state
end

function RLBase.reset!(env::FurutaEnv{A,T}, x0) where {A,T}
    env.state = T.(vec(x0))
    if isdefined(Main, :CuArrays)
        env.state = env.state |> gpu
    end
    env.t = 0
    env.reward = T(0.0)
    env.done = false
    return state
end

function _get_obs(env::FurutaEnv)
    θ, θ̇ , ϕ, ϕ̇ = env.state
    return vcat(sin.(θ), cos.(θ), θ̇ , ϕ, ϕ̇ )
end

angle_normalize(x) = Base.mod((x + Base.π), (2 * Base.π)) - Base.π

function _reward(env::FurutaEnv, u)
    θ, θ̇, ϕ, ϕ̇ = env.state
    r = -(5*angle_normalize(θ)^2 + 0.05*θ̇ ^2 + ϕ^2 + 0.05*ϕ̇ ^2 + 0.05*u^2) - 10000*(abs(ϕ) > 2*π)
    done = (abs(ϕ) > 2*π) #|| (abs(angle_normalize(θ)) < 0.05 && abs(θ̇ ) < 0.5)
    return r, done
end

function _f(env::FurutaEnv, x, u)
    θ, θ̇, ϕ, ϕ̇ = x

    par = env.params
    J,M,ma,mp,la,lp = par.J,par.M,par.ma,par.mp,par.la,par.lp
    g = 9.81
    α = J+(M+ma/3+mp)*la^2
    β = (M+mp/3)*lp^2
    γ = (M+mp/2)*la*lp
    δ = (M+mp/2)*g*lp

    ψ = 1/(α*β - γ^2 + (β^2 + γ^2)*sin(θ)^2)

    dθ = θ̇
    dθ̇ = ψ*(β*(α+β*sin(θ)^2)*cos(θ)*sin(θ)*ϕ̇ ^2
        + 2*β*γ*(1-sin(θ)^2)*sin(θ)*ϕ̇ *θ̇  - γ^2*cos(θ)*sin(θ)*θ̇ ^2
        + δ*(α+β*sin(θ)^2)*sin(θ) - γ*cos(θ)*u)
    dϕ = ϕ̇
    dϕ̇ = ψ*(β*γ*(sin(θ)^2-1)*sin(θ)*ϕ̇ ^2 - 2*β^2*cos(θ)*sin(θ)*ϕ̇ *θ̇
        + β*γ*sin(θ)*θ̇ ^2 - γ*δ*cos(θ)*sin(θ) + β*u)
    return vcat(dθ, dθ̇ , dϕ, dϕ̇ )
end
