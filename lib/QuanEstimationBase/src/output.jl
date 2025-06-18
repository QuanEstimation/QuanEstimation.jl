abstract type AbstractOutput end
abstract type no_save end
abstract type savefile end
abstract type save_reward end

mutable struct Output{S} <: AbstractOutput
    f_list::AbstractVector
    opt_buffer::AbstractVector
    res_file::AbstractVector
    io_buffer::AbstractVector
end

function set_f!(output::AbstractOutput, f::Number)
    append!(output.f_list, f)
end

function set_buffer!(output::AbstractOutput, buffer...)
    output.opt_buffer = [buffer...]
end

function set_io!(output::AbstractOutput, buffer...)
    output.io_buffer = [buffer...]
end

Output{T}(opt::AbstractOpt) where {T} = Output{T}([], [], res_file(opt), [])
Output(opt::AbstractOpt; save::Bool = false) =
    save ? Output{savefile}(opt) : Output{no_save}(opt)

save_type(::Output{savefile}) = :savefile
save_type(::Output{no_save}) = :no_save

function SaveFile(output::Output{no_save}; suffix::AbstractString = ".dat")
    # # JLD2 save
    # open("f.csv", "w") do f
    #     writedlm(f, output.f_list)
    # end

    # CSV save   
    df = DataFrame(f = output.f_list)
    CSV.write("f.csv", df)
    for (res, file) in zip(output.opt_buffer, output.res_file)
        # # JLD2 save
        # jldopen(file * suffix, "w") do f
        #     f[file] = res
        # end

        # CSV save   
        df = DataFrame(res = res)
        CSV.write(file * suffix * ".csv", df)
    end
end

function SaveFile(output::Output{savefile}) end

function SaveCurrent(output::Output{savefile}; suffix::AbstractString = ".dat")
    # # JLD2 save
    # open("f.csv", "a") do f
    #     writedlm(f, output.f_list[end])
    # end

    # CSV save
    df = DataFrame(f = [output.f_list[end]])
    CSV.write("f.csv", df; append=true)
    for (res, file) in zip(output.opt_buffer, output.res_file)
        # # JLD2 save
        # fs = isfile(file * suffix) ? load(file * suffix)[file] : typeof(res)[]
        # jldopen(file * suffix, "w") do f
        #     f[file] = append!(fs, [res])

        # CSV save
        csvfile = file * suffix * ".csv"      
        if isfile(csvfile)
            df_res = CSV.read(csvfile, DataFrame)
        else
            df_res = DataFrame(res = typeof(res)[])
        end
        newrow = DataFrame(res = [res])
        df_res = vcat(df_res, newrow)
        CSV.write(csvfile, df_res)
    end
end

function SaveCurrent(output::Output{no_save}) 
end

function SaveReward(output::Output{savefile}, reward::Number) ## TODO: reset file
    # # JLD2 save
    # open("reward.csv", "a") do r
    #     writedlm(r, reward)
    # end

    # CSV save
    df = DataFrame(reward = [reward])
    CSV.write("reward.csv", df; append=true)
end

function SaveReward(output::Output{no_save}, reward::Number)
end

function SaveReward(rewards)
    # # JLD2 save
    # open("reward.csv", "w") do r
    #     writedlm(r, rewards)
    # end

    # CSV save
    df = DataFrame(reward = rewards)
    CSV.write("reward.csv", df)
end
