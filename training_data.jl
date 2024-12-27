using Pkg
Pkg.activate(".")

using Flux, Statistics, ProgressMeter
using CUDA
using JSON
using CSV
using Sound
using WAV
using DSP
using FFTW
using Random
using CUDA
using Transformers
enable_gpu(CUDA.functional())

using PyPlot; pygui(true)

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

function normalize!(u,fs,target_ampl=0.1)
    env=envelope(u,fs)
    mx=maximum(env)
    gain=target_ampl/mx
    u .*= gain
    return u
end

import DSP.Periodograms.spectrogram

function DSP.Periodograms.spectrogram(X::AbstractMatrix{T}, n::Int=ispow2(size(X,1)) ? size(X,1) : 2size(X,1)-2, noverlap::Int=n>>1;
    onesided::Bool=!ispow2(size(X,1)),
    nfft::Int=nextfastfft(n), 
    fs::Real=1,
    window::Union{Function,AbstractVector,Nothing}=nothing) where {T}

    _, norm2 = DSP.Periodograms.compute_window(window, nfft)
    result=similar(Array{DSP.Util.fftabs2type(T)}, axes(X))  
    r = 1/fs/norm2
    
    if onesided
        for (c,columnX) in enumerate(eachcol(X))
            result[1,c]=abs2(columnX[1])*r
            result[2:end-1,c]=abs2.(columnX[2:end-1])*2r
            result[end,c]=abs2(columnX[end-1])*r
        end
    else
        for (c,columnX) in enumerate(eachcol(X))
            result[:,c]=columnX[:]*r
        end
    end

    DSP.Periodograms.Spectrogram(result, onesided ? rfftfreq(nfft, fs) : fftfreq(nfft, fs),
        (n/2 : n-noverlap : (size(X,2)-1)*(n-noverlap)+n/2) / fs)
end

function mel_filter_bank(num_filters, fft_size, sample_rate, min_freq, max_freq)
    min_mel = 1125.0 * log(1 + min_freq / 700.0)
    max_mel = 1125.0 * log(1 + max_freq / 700.0)

    mel_points = range(min_mel, max_mel, length=num_filters + 2)
    hz_points = 700.0 * (exp.(mel_points ./ 1125.0) .- 1.0)
    bin_points = floor.(hz_points / (sample_rate / fft_size))

    filter_bank = zeros(num_filters, div(fft_size, 2) + 1)
    for i in 1:num_filters
        for j in Int(bin_points[i]):Int(bin_points[i + 1])
            filter_bank[i, j + 1] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
        end
        for j in Int(bin_points[i + 1]):Int(bin_points[i + 2])
            filter_bank[i, j + 1] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])
        end
    end
    return filter_bank
end

function whiten!(x,y)
    offset=mean(x)
    x.-=offset
    y.-=offset
    scale=std(x[:])
    x./=scale
    y./=scale
    return (offset=offset,scale=scale)
end

function clip_noise(X,Y,threshold=-60)
    min_noise=max(maximum(X),maximum(Y))*db2pow(threshold)
    X= X .|> U -> max(U,min_noise)
    Y= Y .|> U -> max(U,min_noise)
    return X,Y
end


time_chain(u, fs, n=length(u)) = normalize!(u,fs) |> 
                u -> u[1:n,1:1] 

freq_chain(U, A) = U |> U -> A*U |>
                U -> pow2db.(U)

power_chain(U, m, fs, w) =  spectrogram(U, m; fs=fs, window=w) |>
        power



function audio_chain(signal, noise, n, w, A)
    m=length(w)

    y=signal
    x=noise
    
    X=stft(x[:].+y[:], m; fs=fs, window=w)
    Y=stft(y[:], m; fs=fs, window=w)
    
    X=power_chain(X, m, fs, w)
    Y=power_chain(Y, m, fs, w)
    
    X,Y = clip_noise(X,Y)
    
    X=freq_chain(X, A)
    Y=freq_chain(Y, A)
    
    whiten!(X,Y)
    return X,Y
end

function overlapping_splits(namedtuple::NamedTuple, args...)
    result=map(t -> overlapping_splits(t, args...),values(namedtuple))
    return NamedTuple{keys(namedtuple)}(result)
end

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
fs=8192
noises=read_noises("./data/4/archive/esc50.csv", "frog", "./data/4/archive/audio/audio/16000/", fs) |> 
    shuffle_n_split |>
    splits -> overlapping_splits(splits, 16384, 12288)

signals=read_signals("./data/fluent_speech_commands_dataset/data/test_data.csv", "./data/fluent_speech_commands_dataset/", "7B4XmNppyrCK977p", fs, n_samples) |>
    shuffle_n_split




function dataloader(signals,noises)
    m=256
    n=16384
    w=hanning(m+1)[2:m+1]
    A=mel_filter_bank(26, m, fs, 300, fs/2)
    
    n_noises=length(noises)
    n_signal=length(signals)
    
    X=Array{Float64, 3}(undef, 26, 127, n_noises*n_signal) |> todevice
    Y=Array{Float64, 3}(undef, 26, 127, n_noises*n_signal) |> todevice
    
    jk=0
    for j in 1:n_signal
        for k in 1:n_noises
            jk += 1
            X[:,:,jk],Y[:,:,jk]=audio_chain(signals[j], noises[k], n, w, A) 
        end
    end
    masks=size(X,2)*ones(Int32,size(X,3)) |> todevice
    
    
    result=Flux.DataLoader((x=X, y=Y, mask=masks); batchsize=10, shuffle=true);
    return result
end


train=dataloader(signals.train,noises.train)
test=dataloader(signals.test,noises.test)



sample=first(train)
sample1=(x=sample.x[:,:,1:1],y=sample.y[:,:,1:1],mask=sample.mask[1:1])
input=withmask(sample1)
input=withmask(sample)
ŷ=model(input.x)


figure(figsize=(10, 5))
imshow(sample1.x[:,:,1], aspect = "auto")
colorbar()


figure(figsize=(10, 5))
imshow(sample1.y[:,:,1], aspect = "auto")
colorbar()


figure(figsize=(10, 5))
imshow(ŷ.hidden_state[:,:,1], aspect = "auto")
colorbar()




figure(figsize=(10, 5))
imshow(X[:,:,10], aspect = "auto")
colorbar()

figure(figsize=(10, 5))
imshow(Y[:,:,10], aspect = "auto")
colorbar()

p=10
frog_sound=frog_sounds[p]
speaker_talk=speaker_talks[p]
sound(speaker_talk, fs)


M=size(X,2)*ones(Int32,size(X,3)) |> todevice

dl=Flux.DataLoader((x=X, y=Y, mask=M); batchsize=10, shuffle=true);


figure(figsize=(10, 5))
imshow(X[:,:,10], aspect = "auto")
colorbar()



sample=first(dl)
input=withmask(sample)
y_=model(input.x)

figure(figsize=(10, 5))
imshow(input.x.hidden_state[:,:,1], aspect = "auto")



figure(figsize=(10, 5))
imshow(input.y[:,:,1], aspect = "auto")

figure(figsize=(10, 5))

]

imshow(y_.hidden_state[:,:,1], aspect = "auto")



sample=first(dl)
sample.x


input=withmask(sample)
model(input.x)

figure(figsize=(10, 5))
withmask(X[:,:,10])


X_=X[:,:,10:10]
Y_=Y[:,:,10:10]
M_=size(X_,2)*ones(Int32,size(X_,3)) |> todevice

Y__hat=model=(withmask((x=X_,y=Y_,mask=M_)))
figure(figsize=(10, 5))
imshow(Y__hat.y, aspect = "auto")

size(Y__hat.y)

figure(figsize=(10, 5))
imshow(Y_, aspect = "auto")
