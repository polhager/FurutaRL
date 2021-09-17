export CircularArraySARTSTrajectory,
    InitStage,
    PreTrainStage,
    INIT_STAGE,
    PRETRAIN_STAGE,
    record!

using ReinforcementLearning
using StatsBase
using CircularArrayBuffers
using Random

#A few extensions to JuliaReinforcementLearningCore that were useful

const CircularArraySARTSTrajectory = Trajectory{
    <:NamedTuple{
        SARTS,
        <:Tuple{
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
        },
    },
}

CircularArraySARTSTrajectory(;
    capacity::Int,
    state = Int => (),
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
    next_state = Int => (),
) = CircularArrayTrajectory(;
    capacity = capacity,
    state = state,
    action = action,
    reward = reward,
    terminal = terminal,
    next_state = next_state,
)

Base.length(t::CircularArraySARTSTrajectory) = length(t[:terminal])

function fetch!(
    s::BatchSampler{SARTS},
    t::CircularArraySARTSTrajectory,
    inds::Vector{Int}
)
    batch = NamedTuple{SARTS}(
        (
            (consecutive_view(t[x], inds) for x in SARTS)...,
        )
    )
    if isnothing(s.cache)
        s.cache = map(batch) do x
            convert(Array, x)
        end
    else
        map(s.cache, batch) do dest, src
            copyto!(dest, src)
        end
    end
end

function StatsBase.sample(rng::AbstractRNG, t::CircularArraySARTSTrajectory, s::BatchSampler)
    inds = rand(rng, 1:length(t), s.batch_size)
    fetch!(s, t, inds)
    inds, s.cache
end

struct InitStage <: AbstractStage end
const INIT_STAGE = InitStage()

struct PreTrainStage <: AbstractStage end
const PRETRAIN_STAGE = PreTrainStage()

function record!(t::AbstractTrajectory, experience::NamedTuple)
    for key in keys(t)
        if haskey(experience, key)
            push!(t[key], experience[key])
        else
            throw(ArgumentError("Missing key in experience: " * key))
        end
    end
end
nothing
