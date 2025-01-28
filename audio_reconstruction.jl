function wrapped_model(U::Matrix)
    (m,n)=size(U)
    sample=(x=reshape(U,m,n,1) |> todevice,y=reshape(U,m,n,1) |> todevice,mask=[size(U,2)] |> todevice) |> todevice
    result=model(withmask(sample).x ).hidden_state |> cpu
    result=reshape(result,size(result,1),size(result,2))
end


function antiwhiten!(x,white)
    x.*=white.scale
    x.+=white.offset
    return x
end

fullcircle(X)=vcat(X,conj.(X[end-1:-1:2]))

function antistft(X)
    m=2(size(X,1)-1)
    n=size(X,2)
    l=m*n-(n-1)*m>>1
    result=zeros(Complex,l)
    p=1
    for k in 1:n
        result[p:p+m-1]+=ifft(fullcircle(X[:,k]))
        p+=m>>1
    end
    return result
end


regular_columns(W)=[~iszero(c) for c in eachcol(W)] |> BitVector

function attenuations(X2,MY2,M)
    attenuations=zeros(eltype(X2),size(X2))
    rc=regular_columns(M)
    M=view(M,:,rc)
    X2=view(X2,rc,:)
    for p in 1:size(X2,2)
        u=X2[:,p]
        y=MY2[:,p]
        w=(M*Diagonal(u)*M')\y
        attenuation2=M'*w
        clamp!(attenuation2, 0, 1)
        attenuations[rc,p]=sqrt.(attenuation2)
    end
    return attenuations
end

function withintermediate(model,fs,win,u,white=nothing)
    m=length(win)
    X=stft(u[:], m; fs=fs, window=win)
    X2=spectrogram(X, m; fs=fs, window=win) |> power
    MX2=power_chain(X, m, fs, win)|> clip_noise |> U -> freq_chain(U, M)
    
    whiteMX2=copy(MX2)
    white=whiten!(whiteMX2,white)
    
    whiteMY2=wrapped_model(whiteMX2)
    MY2=copy(whiteMY2)
    antiwhiten!(MY2,white)
    MY2=db2pow.(MY2) 

    att=ones(eltype(X2),size(X2))
    try
        att=attenuations(X2,MY2,M)
    catch e
        nothing
    end
    
    y=real.(antistft(att .* X))
    return y,X,X2,MX2,whiteMX2,whiteMY2,white,att
end

function suppressnoise(model,fs,win,u)
    y,_,_,_,_,_,_,_=withintermediate(model,fs,win,u)
    return y
end

signal=signals.test[3]
noise=noises.test[9]

u=signal+noise

_,_,_,_,whiteMX2,whiteMY2,_,_=withintermediate(model,fs,win,signal)

figure(figsize=(10, 5))
imshow(whiteMX2, aspect = "auto")
title("Signal without Noise")
colorbar()
savefig("signal.png")

_,_,_,_,whiteMX2,whiteMY2,_,_=withintermediate(model,fs,win,u)


figure(figsize=(10, 5))
imshow(whiteMY2, aspect = "auto")
title("Cleaned Signal with Noise")
colorbar()
savefig("result.png")

figure(figsize=(10, 5))
imshow(whiteMX2, aspect = "auto")
title("Signal with Noise")
colorbar()
savefig("with_noise.png")

y=suppressnoise(model,fs,win,u)

wavwrite(vcat(y,zeros(fs)),"result.wav";Fs=8192)
wavwrite(vcat(u,zeros(fs)),"with_noise.wav";Fs=8192)
wavwrite(vcat(signal,zeros(fs)),"signal.wav";Fs=8192)

sound(y, fs) # 
sound(u, fs)
