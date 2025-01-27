
function envelope(u, fs, τ_attack=0.01, τ_decay=0.1)
    α_attack = exp(-1 / (fs * τ_attack))
    α_decay = exp(-1 / (fs * τ_decay))
    
    abs_u = abs.(hilbert(u))
    result=similar(abs_u)
    result[1]=0
    
    for k in 2:length(result)
        α=abs_u[k] > result[k-1] ? α_attack : α_decay
        result[k]= α * abs_u[k-1] + (1 - α) * abs_u[k]
    end
    return result
end

function normalize!(u,fs,target_ampl=0.25)
    env=envelope(u,fs)
    mx=maximum(env)
    gain=target_ampl/mx
    u .*= gain
    return u
end

time_chain(u, fs, n=length(u)) = normalize!(u,fs) |> 
                u -> u[1:n,1:1] 

function read_signals(data_description, path, speaker, fs, n_samples)
    signal_files=CSV.read(data_description, identity)
    filtered_files=signal_files.path[signal_files.speakerId .== speaker]
    
    signals=Vector{Matrix{Float64}}()
    for filename in filtered_files
        u,fs_in=wavread(path*filename)
        if length(u) >= 2fs_in
            u=resample(u,fs/fs_in;dims=1) 
            u=time_chain(u, fs, n_samples)
            push!(signals,u)
        end
    end
    return signals
end

function shuffle_n_split(data, testpercentage=10)
    n=length(data)
    p=randperm(n)
    permute!(data,p)
    split=floor(Int64,n*testpercentage/100)
    return (all=data, train=view(data,split+1:n), test=view(data,1:split))
end

function overlapping_splits(v::AbstractVector, n::Int, step::Int)
    result=Vector{eltype(v)}()
    for elem in v
        thru=n 
        while thru<=length(elem)
            from=thru-n+1
            push!(result,view(elem,from:thru,1:1))
            thru+=step
        end
    end
    return result
end

function overlapping_splits(namedtuple::NamedTuple, args...)
    result=map(t -> overlapping_splits(t, args...),values(namedtuple))
    return NamedTuple{keys(namedtuple)}(result)
end

function read_noises(data_description, type, path, fs)
    noises=Vector{Matrix{Float64}}()
    noise_files=CSV.read(data_description, identity)
    p=findfirst(s -> s==type, noise_files.category)
    t=noise_files.target[p]
    filtered_noise_files=noise_files.filename[noise_files.target .== t]
    for filename in filtered_noise_files
        u,fs_in = wavread(path*filename)
        u= resample(u, fs/fs_in; dims=1) 
        u= time_chain(u, fs)
        push!(noises,u)
    end
    return noises
end

n_samples=16384
stepsize=12288
fs=8192

noises=read_noises("./data/environmental_sound_classification_50/archive/esc50.csv", "frog", "./data/environmental_sound_classification_50/archive/audio/audio/16000/", fs) |> 
    shuffle_n_split |>
    splits -> overlapping_splits(splits, n_samples, stepsize)

signals=read_signals("./data/fluent_speech_commands_dataset/data/test_data.csv", "./data/fluent_speech_commands_dataset/", "7B4XmNppyrCK977p", fs, n_samples) |>
    shuffle_n_split



