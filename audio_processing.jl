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

function filter_bank(num_filters, fft_size, args...)
    if num_filters == (fft_size ÷ 2)+1
        result=Diagonal(ones(num_filters))
    else
        result=mel_filter_bank(num_filters, args...)
    end
    return result
end

function whiten!(x::Matrix,y::Matrix)
    offset=mean(x)
    x.-=offset
    y.-=offset
    scale=std(x[:])
    x./=scale
    y./=scale
    return (offset=offset,scale=scale)
end

function whiten!(x::Matrix,white)
    x.-=white.offset
    x./=white.scale
    return white
end

function whiten!(x::Matrix, w::Nothing)
    offset=mean(x)
    x.-=offset
    scale=std(x[:])
    x./=scale
    return (offset=offset,scale=scale)
end

function clip_noise(X::AbstractMatrix, threshold=-60)
    min_noise=maximum(X)*db2pow(threshold)
    X= X .|> U -> max(U,min_noise)
    return X
end

function clip_noise(X::AbstractMatrix,Y::AbstractMatrix, threshold=-60)
    min_noise=max(maximum(X),maximum(Y))*db2pow(threshold)
    X= X .|> U -> max(U,min_noise)
    Y= Y .|> U -> max(U,min_noise)
    return X,Y
end

freq_chain(U, A) = U |> U -> A*U |>
                U -> pow2db.(U)

power_chain(U, m, fs, w) =  spectrogram(U, m; fs=fs, window=w) |>
        power

function audio_chain(signal, noise, w, A)
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

function dataloader(signals,noises,win,M; kwargs...)
    m=length(win)
    d_in=size(M,1)
    n_samples=length(signals[1])
    n_steps=2(n_samples ÷ m)-1

    n_noises=length(noises)
    n_signal=length(signals)

    X=Array{Float64, 3}(undef, d_in, n_steps, n_noises*n_signal) |> todevice
    Y=Array{Float64, 3}(undef, d_in, n_steps, n_noises*n_signal) |> todevice
    
    jk=0
    for j in 1:n_signal
        for k in 1:n_noises
            jk += 1
            X[:,:,jk],Y[:,:,jk]=audio_chain(signals[j], noises[k], win, M) 
        end
    end
    
    masks=size(X,2)*ones(Int32,size(X,3)) |> todevice
    
    result=Flux.DataLoader((x=X, y=Y, mask=masks); kwargs...);
    return result
end

win=hanning(256+1)[2:end]
m=length(win)

f_low=300
f_high=fs>>1
M=filter_bank(d_in, m, fs, f_low, f_high)

batchsize=32

train=dataloader(signals.train,noises.train, win, M; batchsize=batchsize, shuffle=true)
test=(oos=dataloader(signals.test,noises.test, win, M; batchsize=batchsize, shuffle=true),
      is= dataloader(signals.train[1:length(signals.test)], noises.train[1:length(noises.test)], win, M; batchsize=batchsize, shuffle=true))
