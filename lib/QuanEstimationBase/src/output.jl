"""
    AbstractOutput

Abstract supertype for the I/O subsystem. The type hierarchy is:

- `AbstractOutput`
  - `Output{no_save}`: No output saved.
  - `Output{savefile}`: Full optimization history saved.
  - `Output{save_reward}`: Only reward values saved.

See also: [`Output`](@ref), [`no_save`](@ref), [`savefile`](@ref), [`save_reward`](@ref).
"""
abstract type AbstractOutput end
"""
    no_save

Dispatch label type indicating that no output should be saved during optimization.

See also: [`savefile`](@ref), [`save_reward`](@ref).
"""
abstract type no_save end
"""
    savefile

Dispatch label type indicating that the full optimization history should be saved to files.

See also: [`no_save`](@ref), [`save_reward`](@ref).
"""
abstract type savefile end
"""
    save_reward

Dispatch label type indicating that only reward (objective function) values should be saved.

See also: [`no_save`](@ref), [`savefile`](@ref).
"""
abstract type save_reward end

"""
    Output{S}

Mutable struct holding optimization output data.

**Fields:**
- `f_list`: History of objective function values.
- `opt_buffer`: Buffer for optimization results.
- `res_file`: File names for saving results.
- `io_buffer`: Buffer for I/O messages.
"""
mutable struct Output{S} <: AbstractOutput
    f_list::AbstractVector
    opt_buffer::AbstractVector
    res_file::AbstractVector
    io_buffer::AbstractVector
end

"""

    set_f!(output::AbstractOutput, f::Number)

Append an objective function value to the output history.
"""
function set_f!(output::AbstractOutput, f::Number)
    append!(output.f_list, f)
end

"""

    set_buffer!(output::AbstractOutput, buffer...)

Set the optimization result buffer.
"""
function set_buffer!(output::AbstractOutput, buffer...)
    output.opt_buffer = [buffer...]
end

"""

    set_io!(output::AbstractOutput, buffer...)

Set the I/O message buffer.
"""
function set_io!(output::AbstractOutput, buffer...)
    output.io_buffer = [buffer...]
end

"""
    Output{T}(opt::AbstractOpt)

Construct a generic `Output` container with type parameter `T` (e.g., `no_save`, `savefile`).

Initializes empty buffers and retrieves result file names from `opt`.
"""
Output{T}(opt::AbstractOpt) where {T} = Output{T}([], [], res_file(opt), [])
"""
    Output(opt::AbstractOpt; save=false)

Convenience constructor for `Output`.

- If `save = true`, constructs `Output{savefile}` (full history saved).
- If `save = false` (default), constructs `Output{no_save}` (nothing saved).
"""
Output(opt::AbstractOpt; save::Bool = false) =
    save ? Output{savefile}(opt) : Output{no_save}(opt)

"""
    save_type(::Output{savefile})

Return the symbol `:savefile`.
"""
save_type(::Output{savefile}) = :savefile
"""
    save_type(::Output{no_save})

Return the symbol `:no_save`.
"""
save_type(::Output{no_save}) = :no_save

"""

    SaveFile(output::Output{no_save}, suffix=".dat")

Save the full optimization history to JLD2 files. No-op for `savefile` type outputs.
"""
function SaveFile(output::Output{no_save}, suffix::AbstractString = ".dat")
    # JLD2 save
    open("f.csv", "w") do f
        writedlm(f, output.f_list)
    end

    # # CSV save   
    # df = DataFrame(f = output.f_list)
    # CSV.write("f.csv", df)
    for (res, file) in zip(output.opt_buffer, output.res_file)
        # JLD2 save (atomic: write temp → rename)
        tmp = tempname(pwd())
        jldopen(tmp, "w") do f
            f[file] = res
        end
        mv(tmp, file * suffix; force=true)

        # # CSV save   
        # df = DataFrame(res = res)
        # CSV.write(file * ".csv", df)
    end
end

"""
    SaveFile(output::Output{savefile})

No-op: `savefile`-type outputs do not support full-history save via `SaveFile`.

See [`SaveFile(::Output{no_save})`](@ref) for the active implementation.
"""
function SaveFile(output::Output{savefile}) end

"""

    SaveCurrent(output::Output{savefile}, suffix=".dat")

Append the current optimization result to the save file. No-op for `no_save` type outputs.
"""
function SaveCurrent(output::Output{savefile}, suffix::AbstractString = ".dat")
    # JLD2 save
    open("f.csv", "a") do f
        writedlm(f, output.f_list[end])
    end

    # # CSV save
    # df = DataFrame(f = [output.f_list[end]])
    # CSV.write("f.csv", df; append=true)
    for (res, file) in zip(output.opt_buffer, output.res_file)
        # JLD2 save (atomic: read target → write temp → rename)
        target = file * suffix
        fs = isfile(target) ? load(target)[file] : typeof(res)[]
        tmp = tempname(pwd())
        jldopen(tmp, "w") do f
            f[file] = append!(fs, [res])
        end
        mv(tmp, target; force=true)
    end
end

"""
    SaveCurrent(output::Output{no_save})

No-op: `no_save`-type outputs do not save current results.

See [`SaveCurrent(::Output{savefile})`](@ref) for the active implementation.
"""
function SaveCurrent(output::Output{no_save}) end

"""

    SaveReward(output, reward)

Save a reward value to `reward.csv`. For `no_save` outputs, this is a no-op. When called with a vector, writes all rewards at once.
"""
function SaveReward(output::Output{savefile}, reward::Number)
    # JLD2 save
    open("reward.csv", "a") do r
        writedlm(r, reward)
    end

    # # CSV save
    # df = DataFrame(reward = [reward])
    # CSV.write("reward.csv", df; append=true)
end

"""
    SaveReward(output::Output{no_save}, reward::Number)

No-op: `no_save`-type outputs do not save reward values.

See [`SaveReward(::Output{savefile}, ::Number)`](@ref) for the active implementation.
"""
function SaveReward(output::Output{no_save}, reward::Number) end

"""
    SaveReward(rewards)

Convenience method: write a vector of reward values to `reward.csv` at once.
"""
function SaveReward(rewards)
    # JLD2 save
    open("reward.csv", "w") do r
        writedlm(r, rewards)
    end    
end
