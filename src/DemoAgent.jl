export DemoAgent, action

using ReinforcementLearning
import Functors: functor
using Setfield: @set

Base.@kwdef struct DemoAgent{P<:AbstractPolicy, T<:AbstractTrajectory} <: AbstractPolicy
    policy::P
    trajectory::T
    demo_trajectory::T
end

functor(x::DemoAgent) = (policy = x.policy,), y -> @set x.policy = y.policy

(agent::DemoAgent)(env) = agent.policy(env)

function check(agent::DemoAgent, env::AbstractEnv)
    if ActionStyle(env) === FULL_ACTION_SET &&
        !haskey(agent.trajectory, :legal_actions_mask)
        !haskey(agent.demo_trajectory, :legal_actions_mask)
    end
    check(agent.policy, env)
end

function action(agent::DemoAgent, s; training = true)
    action(agent.policy, s, training=training)
end

function RLBase.update!(agent::DemoAgent)
    if !isnothing(agent.demo_trajectory) && length(agent.demo_trajectory) > 0
        update!(agent.policy, agent.trajectory, agent.demo_trajectory)
    else
        update!(agent.policy, agent.trajectory)
    end
end


Base.nameof(agent::DemoAgent) = nameof(agent.policy)

#####
# Default behaviors
#####

"""
Here we extend the definition of `(p::AbstractPolicy)(::AbstractEnv)` in
`RLBase` to accept an `AbstractStage` as the first argument. Algorithm designers
may customize these behaviors respectively by implementing:
- `(p::YourPolicy)(::AbstractStage, ::AbstractEnv)`
- `(p::YourPolicy)(::PreActStage, ::AbstractEnv, action)`
The default behaviors for `Agent` are:
1. Update the inner `trajectory` given the context of `policy`, `env`, and
   `stage`.
  1. By default we do nothing.
  2. In `PreActStage`, we `push!` the current **state** and the **action** into
     the `trajectory`.
  3. In `PostActStage`, we query the `reward` and `is_terminated` info from
     `env` and push them into `trajectory`.
  4. In the `PostEpisodeStage`, we push the `state` at the end of an episode and
     a dummy action into the `trajectory`.
  5. In the `PreEpisodeStage`, we pop out the lastest `state` and `action` pair
     (which are dummy ones) from `trajectory`.
2. Update the inner `policy` given the context of `trajectory`, `env`, and
   `stage`.
  1. By default, we only `update!` the `policy` in the `PreActStage`. And it's
     despatched to `update!(policy, trajectory)`.
"""

function (agent::DemoAgent)(stage::AbstractStage, env::AbstractEnv)
    update!(agent.trajectory, agent.policy, env, stage)
    update!(agent.policy, agent.trajectory, agent.demo_trajectory, env, stage)
end

function (agent::DemoAgent)(stage::PreActStage, env::AbstractEnv, action)
    update!(agent.trajectory, agent.policy, env, stage, action)
    update!(agent.policy, agent.trajectory, agent.demo_trajectory)
end

function (agent::DemoAgent)(stage::AbstractStage, stage2::InitStage, env::AbstractEnv)
    update!(agent.trajectory, agent.policy, env, stage)
    update!(agent.demo_trajectory, agent.policy, env, stage)
end

function (agent::DemoAgent)(stage::AbstractStage, stage2::PreTrainStage, env::AbstractEnv)
    #update!(agent.policy, agent.demo_trajectory, env, stage)
end

function (agent::DemoAgent)(stage::PreActStage, stage2::InitStage, env::AbstractEnv, action)
    update!(agent.trajectory, agent.policy, env, stage, action)
    update!(agent.demo_trajectory, agent.policy, env, stage, action)

end

function (agent::DemoAgent)(stage::PreTrainStage, env::AbstractEnv)
    update!(agent.policy, agent.demo_trajectory, env, stage)
end

function RLBase.update!(
    ::AbstractPolicy,
    ::AbstractTrajectory,
    ::AbstractTrajectory,
    ::AbstractEnv,
    ::AbstractStage,
    ::AbstractStage,
) end

function RLBase.update!(
    ::AbstractPolicy,
    ::AbstractTrajectory,
    ::AbstractTrajectory,
    ::AbstractEnv,
    ::AbstractStage,
) end

#####
# Default behaviors for known trajectories
#####

function RLBase.update!(
    ::AbstractTrajectory,
    ::AbstractPolicy,
    ::AbstractEnv,
    ::AbstractStage,
    ::AbstractStage,
) end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    demo_trajectory::AbstractTrajectory,
    ::AbstractPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    if length(trajectory) > 0
        pop!(trajectory[:state])
        pop!(trajectory[:action])
        if haskey(trajectory, :legal_actions_mask)
            pop!(trajectory[:legal_actions_mask])
        end
    end

    if length(demo_trajectory) > 0
        pop!(demo_trajectory[:state])
        pop!(demo_trajectory[:action])
        if haskey(demo_trajectory, :legal_actions_mask)
            pop!(demo_trajectory[:legal_actions_mask])
        end
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    demo_trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)
    push!(trajectory[:state], s)
    push!(trajectory[:action], action)
    if haskey(trajectory, :legal_actions_mask)
        lasm = policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) : legal_action_space_mask(env)
        push!(trajectory[:legal_actions_mask], lasm)
    end

    push!(demo_trajectory[:state], s)
    push!(demo_trajectory[:action], action)
    if haskey(demo_trajectory, :legal_actions_mask)
        lasm = policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) : legal_action_space_mask(env)
        push!(demo_trajectory[:legal_actions_mask], lasm)
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    demo_trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
    # Note that for trajectories like `CircularArraySARTTrajectory`, data are
    # stored in a SARSA format, which means we still need to generate a dummy
    # action at the end of an episode. Here we simply select a random one using
    # the global rng. In theory it shouldn't affect the performance of specific
    # algorithm.
    # TODO: how to inject a local rng here to avoid polluting the global rng
    action = rand(action_space(env))

    s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)
    push!(trajectory[:state], s)
    push!(trajectory[:action], action)
    if haskey(trajectory, :legal_actions_mask)
        lasm = policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) : legal_action_space_mask(env)
        push!(trajectory[:legal_actions_mask], lasm)
    end

    push!(demo_trajectory[:state], s)
    push!(demo_trajectory[:action], action)
    if haskey(demo_trajectory, :legal_actions_mask)
        lasm = policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) : legal_action_space_mask(env)
        push!(demo_trajectory[:legal_actions_mask], lasm)
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    demo_trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    r = policy isa NamedPolicy ? reward(env, nameof(policy)) : reward(env)
    push!(trajectory[:reward], r)
    push!(trajectory[:terminal], is_terminated(env))

    push!(demo_trajectory[:reward], r)
    push!(demo_trajectory[:terminal], is_terminated(env))
end
