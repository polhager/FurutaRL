using ReinforcementLearning
using Flux
using CUDA
using Random, Distributions
using Setfield: @set
using Zygote: ignore
using LinearAlgebra

export DDPGPolicyDemo, update_critic!

mutable struct DDPGPolicyDemo{
    BA<:NeuralNetworkApproximator,
    BC<:NeuralNetworkApproximator,
    TA<:NeuralNetworkApproximator,
    TC<:NeuralNetworkApproximator,
    P,
    R<:AbstractRNG,
} <: AbstractPolicy

    behavior_actor::BA
    behavior_critic::BC
    target_actor::TA
    target_critic::TC
    γ::Float32
    ρ::Float32
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_every::Int
    act_limit::Float64
    act_noise::Float64
    step::Int
    rng::R
    # for logging
    actor_loss::Float32
    critic_loss::Float32
end

Flux.functor(x::DDPGPolicyDemo) = (
    ba = x.behavior_actor,
    bc = x.behavior_critic,
    ta = x.target_actor,
    tc = x.target_critic,
),
y -> begin
    x = @set x.behavior_actor = y.ba
    x = @set x.behavior_critic = y.bc
    x = @set x.target_actor = y.ta
    x = @set x.target_critic = y.tc
    x
end

"""
    DDPGPolicy(;kwargs...)
# Keyword arguments
- `behavior_actor`,
- `behavior_critic`,
- `target_actor`,
- `target_critic`,
- `start_policy`,
- `γ = 0.99f0`,
- `ρ = 0.995f0`,
- `batch_size = 32`,
- `start_steps = 10000`,
- `update_after = 1000`,
- `update_every = 50`,
- `act_limit = 1.0`,
- `act_noise = 0.1`,
- `step = 0`,
- `rng = Random.GLOBAL_RNG`,
"""
function DDPGPolicyDemo(;
    behavior_actor,
    behavior_critic,
    target_actor,
    target_critic,
    start_policy,
    γ = 0.99f0,
    ρ = 0.995f0,
    batch_size = 32,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    act_limit = 1.0,
    act_noise = 0.1,
    step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(behavior_actor, target_actor)  # force sync
    copyto!(behavior_critic, target_critic)  # force sync
    DDPGPolicyDemo(
        behavior_actor,
        behavior_critic,
        target_actor,
        target_critic,
        γ,
        ρ,
        batch_size,
        start_steps,
        start_policy,
        update_after,
        update_every,
        act_limit,
        act_noise,
        step,
        rng,
        0.0f0,
        0.0f0,
    )
end

function (p::DDPGPolicyDemo)(env::AbstractEnv; training = true)
    p.step += 1

    if p.step <= p.start_steps
        p.start_policy(env)
    else
        D = device(p.behavior_actor)
        s = state(env)
        s = Flux.unsqueeze(s, ndims(s) + 1)
        a = p.behavior_actor(send_to_device(D, s)) |> vec |> send_to_host
        clamp(a[] + randn(p.rng) * p.act_noise * training, -p.act_limit, p.act_limit)
    end
end

function action(p::DDPGPolicyDemo, s; training = true)
    p.step += 1

    if p.step <= p.start_steps
        p.start_policy(s)
    else
        D = device(p.behavior_actor)
        s = Flux.unsqueeze(s, ndims(s) + 1)
        a = p.behavior_actor(send_to_device(D, s)) |> vec |> send_to_host
        clamp(a[] + randn(p.rng) * p.act_noise * training, -p.act_limit, p.act_limit)
    end
end

function RLBase.update!(p::DDPGPolicyDemo, traj::CircularArraySARTSTrajectory, ::AbstractEnv, ::PreActStage)
    length(traj) > p.update_after || return
    p.step % p.update_every == 0 || return
    inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(p.batch_size))
    update!(p, batch)
end

function update_critic!(p::DDPGPolicyDemo, traj::CircularArraySARTSTrajectory, demo_traj::CircularArraySARTSTrajectory)
    inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(ceil(Int, 3*p.batch_size/4)))
    d_inds, d_batch = sample(p.rng, traj, BatchSampler{SARTS}(ceil(Int, p.batch_size/4)))
    _update_critic(p, batch, d_batch=d_batch)
end

function update_critic!(p::DDPGPolicyDemo, traj::CircularArraySARTSTrajectory)
    inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(ceil(Int, p.batch_size)))
    _update_critic(p, batch)
end

function RLBase.update!(p::DDPGPolicyDemo, traj::CircularArraySARTSTrajectory)
    length(traj) > p.update_after || return
    p.step % p.update_every == 0 || return
    inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(p.batch_size))
    update!(p, batch)
end

function RLBase.update!(p::DDPGPolicyDemo, traj::CircularArraySARTSTrajectory, ::AbstractEnv, stage::PreTrainStage)
    inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(p.batch_size))
    update!(p, batch, stage)
end

function RLBase.update!(p::DDPGPolicyDemo, traj::CircularArraySARTSTrajectory, demo_traj::CircularArraySARTSTrajectory)
    length(traj) > p.update_after || return
    p.step % p.update_every == 0 || return
    inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(ceil(Int, p.batch_size*0.75)))
    demo_inds, demo_batch = sample(p.rng, demo_traj, BatchSampler{SARTS}(ceil(Int, p.batch_size*0.25)))
    update!(p, batch, demo_batch)
end

function RLBase.update!(p::DDPGPolicyDemo, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = send_to_device(device(p), batch)

    A = p.behavior_actor
    C = p.behavior_critic
    Aₜ = p.target_actor
    Cₜ = p.target_critic

    γ = p.γ
    ρ = p.ρ

    a′ = Aₜ(s′)
    qₜ = Cₜ(vcat(s′, a′)) |> vec
    y = r .+ γ .* (1 .- t) .* qₜ
    a = Flux.unsqueeze(a, 1)

    gs1 = gradient(Flux.params(C)) do
        q = C(vcat(s, a)) |> vec
        loss = mean((y .- q) .^ 2)
        ignore() do
            p.critic_loss = loss
        end
        loss
    end

    update!(C, gs1)

    gs2 = gradient(Flux.params(A)) do
        loss = -mean(C(vcat(s, A(s))))
        ignore() do
            p.actor_loss = loss
        end
        loss
    end

    update!(A, gs2)

    # polyak averaging
    for (dest, src) in zip(Flux.params([Aₜ, Cₜ]), Flux.params([A, C]))
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end

function RLBase.update!(p::DDPGPolicyDemo, batch::NamedTuple{SARTS}, ::PreTrainStage)
    s, a, r, t, s′ = send_to_device(device(p), batch)

    A = p.behavior_actor
    C = p.behavior_critic
    Aₜ = p.target_actor
    Cₜ = p.target_critic

    γ = p.γ
    ρ = p.ρ

    a′ = Aₜ(s′)
    qₜ = Cₜ(vcat(s′, a′)) |> vec
    y = r .+ γ .* (1 .- t) .* qₜ
    a = Flux.unsqueeze(a, 1)

    gs1 = gradient(Flux.params(C)) do
        q = C(vcat(s, a)) |> vec
        loss = mean((y .- q) .^ 2) + sum(norm, Flux.params(C))/2
        ignore() do
            p.critic_loss = loss
        end
        loss
    end

    update!(C, gs1)

    gs2 = gradient(Flux.params(A)) do
        loss = -mean(C(vcat(s, A(s)))) + 1/2*mean((A(s) .- a) .^ 2) + sum(norm, Flux.params(A))/4
        ignore() do
            p.actor_loss = loss
        end
        loss
    end

    update!(A, gs2)

    # polyak averaging
    for (dest, src) in zip(Flux.params([Aₜ, Cₜ]), Flux.params([A, C]))
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end

function RLBase.update!(p::DDPGPolicyDemo, batch::NamedTuple{SARTS}, demo_batch::NamedTuple{SARTS})
    s, a, r, t, s′ = send_to_device(device(p), batch)
    sᵈ, aᵈ, rᵈ, tᵈ, s′ᵈ = send_to_device(device(p), demo_batch)

    A = p.behavior_actor
    C = p.behavior_critic
    Aₜ = p.target_actor
    Cₜ = p.target_critic

    γ = p.γ
    ρ = p.ρ

    a′ = Aₜ(s′)
    qₜ = Cₜ(vcat(s′, a′)) |> vec
    y = r .+ γ .* (1 .- t) .* qₜ
    a = Flux.unsqueeze(a, 1)

    gs1 = gradient(Flux.params(C)) do
        q = C(vcat(s, a)) |> vec
        loss = mean((y .- q) .^ 2) + sum(norm, Flux.params(C))/2
        ignore() do
            p.critic_loss = loss
        end
        loss
    end

    update!(C, gs1)

    aᵈ = Flux.unsqueeze(aᵈ, 1)
    Qcomp = Cₜ(vcat(sᵈ, aᵈ)) .> Cₜ(vcat(sᵈ, A(sᵈ)))
    gs2 = gradient(Flux.params(A)) do
        loss = -mean(C(vcat(s, A(s)))) .+ 1/2*mean(((A(sᵈ) .- aᵈ) .^ 2) .* Qcomp) .+ sum(norm, Flux.params(A)) ./ 4
        ignore() do
            p.actor_loss = loss
        end
        loss
    end

    update!(A, gs2)

    # polyak averaging
    for (dest, src) in zip(Flux.params([Aₜ, Cₜ]), Flux.params([A, C]))
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end

# gradient gives error: scalar getindex is disallowed if x is a CuArray with Float64, not FLoat32.
# Solved by calling |> gpu at the end, converting CuArray{Float64,n} to CuArray{Float32,n}
function _update_critic(p::DDPGPolicyDemo, batch::NamedTuple{SARTS}; d_batch = nothing)
    s, a, r, t, s′ = send_to_device(device(p), batch)
    if !isnothing(d_batch)
        sᵈ, aᵈ, rᵈ, tᵈ, s′ᵈ = send_to_device(device(p), d_batch)

        s = hcat(s,sᵈ)
        a = vcat(a,aᵈ)
        r = vcat(r,rᵈ)
        t = vcat(t,tᵈ)
        s′ = hcat(s′,s′ᵈ)
    end

    A = p.behavior_actor
    C = p.behavior_critic
    Aₜ = p.target_actor
    Cₜ = p.target_critic

    γ = p.γ
    ρ = p.ρ

    a′ = Aₜ(s′)
    qₜ = Cₜ(vcat(s′, a′)) |> vec
    y = r .+ γ .* (1 .- t) .* qₜ
    a = Flux.unsqueeze(a, 1)

    gs1 = gradient(Flux.params(C)) do
        q = C(vcat(s, a)) |> vec
        loss = mean((y .- q) .^ 2)
        ignore() do
            p.critic_loss = loss
        end
        loss
    end

    update!(C, gs1)

    for (dest, src) in zip(Flux.params(Cₜ), Flux.params(C))
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end


function _pgd(A, C, x, ϵ)
    x0 = copy(x)
    α = 2.5*ϵ/10
    for k in 1:10
        grad = gradient(Flux.params(x)) do
            -mean(C(vcat(x, A(x))))
        end

        x = x + α*sign.(grad[x])
        x = clamp.(x, x0.-ϵ, x0.+ϵ)
        x = send_to_device(device(A), x)
    end
    x
end
