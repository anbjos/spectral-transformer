# Signals and Noises

signals
noises

# Define various parameters

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


# Define and process time domain signal

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

    println("hi")
    
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

u=signals.test[3]+noises.test[1]
y=suppressnoise(model,fs,win,u)

sound(y, fs)
sound(u, fs)



withintermediate(model,fs,win,signals.test[1]+noises.test[2])
withintermediate(model,fs,win,signals.test[5])


####################################
# Check if performance is better with phase defined from the signal input
# it is not, so phase quality reduction from frogs is not degrading performance much
# Conclusion: the degradation in quality is due to imperfection in the attenuation

u=signals.test[3]
uu=u+noises.test[1]

_,ref,_,_,_,_,_,att=withintermediate(model,fs,win,u)
ph=copy(ref)
ph[abs.(ph) .< 256eps(eltype(real.(ph)))] .= 1

y,X,X2,MX2,whiteMX2,whiteMY2,white,att=withintermediate(model,fs,win,uu)

X=abs.(X) .* (ph./(abs.(ph)))
yy=real.(antistft(att .* X))

sound(y, fs)
sound(yy, fs)
sound(u, fs)
sound(uu, fs)

btt=abs.(ref) ./ abs.(X)
clamp!(btt,0.01,1)
yyy=real.(antistft(btt .* X))
sound(yyy, fs)



clf()
p=80
plot(att[:,p])
plot(btt[:,p])
title("ideal att not very smooth")

clf()
plot((M*abs2.(att .* X))[:,p])
plot((M*abs2.(btt .* X))[:,p])





#NOT working (trying to smooth out att)
using DSP
for c in 1:127
    att[:,c]=DSP.conv([1,1,1,1,1],att[:,c])[3:end-2]
end






###################################################
# Check if performance is reduced due to the dimention reduction
# Use ideal model result (=signal without noise) to define attenuations
# Run with d_in as free parameter

d_in=52

ps,pn=5,4

M=mel_filter_bank(d_in, m, fs, f_low, f_high)

x=signals.test[ps]+noises.test[pn]
X=stft(x[:], m; fs=fs, window=win)
X2=spectrogram(X, m; fs=fs, window=win) |> power
MX2=power_chain(X, m, fs, win)|> clip_noise |> U -> freq_chain(U, M)

whiteMX2=copy(MX2)
white=whiten!(whiteMX2,nothing)

x_=signals.test[ps]
X_=stft(x_[:], m; fs=fs, window=win)
X2_=spectrogram(X_, m; fs=fs, window=win) |> power
MX2_=power_chain(X_, m, fs, win)|> clip_noise |> U -> freq_chain(U, M)

MY2=MX2_

MY2=db2pow.(MY2) 

rc=regular_columns(M)

X2=view(X2,rc,:)
M=view(M,:,rc)


att=zeros(eltype(X),size(X))


for p in 1:size(X,2)
    u=X2[:,p]
    y=MY2[:,p]
    w=(M*Diagonal(u)*M')\y
    attenuation2=M'*w
    clamp!(attenuation2, 0, 1)
    att[rc,p]=sqrt.(attenuation2)
end


R=att .* X

y=real.(antistft(R))

sound(y, fs)
sound(x, fs)


sound(signals.test[ps],fs)

###############################

figure(figsize=(10, 5))
imshow(whiteMX2, aspect = "auto")
title("raw whiteMX2")
colorbar()




figure(figsize=(10, 5))
imshow(whiteY2, aspect = "auto")
title("clean whiteY2")
colorbar()


#Preserves the information from whiteY2.

println("output to input power ratio: $(sum(Y2)/sum(db2pow.(MX2)))")

# select a sftf and compare input and output power spectrum
p=45
clf()
plot(Y2[:,p])
plot(db2pow.(MX2[:,p]))
title("Linear Power ")
xlabel("Frequency, MEL logarithmic")
ylabel("Power")

ignored_columns(W)=[iszero(c) for c in eachcol(W)] |> BitVector
ic=ignored_columns(M)

U0=X2[:,p][.~ic]
M0=M[:,.~ic]
Y0=M0*U0
Y1=Y2[:,p]

x=(M0*Diagonal(U0)*M0')\Y1

#Verify match att*X2 to Y2 in MEL domain. THIS MATCHES
figure()
#plot(M0*Diagonal(M0'*x)*U0)
plot(Y1)     # => Y1=M0*Diagonal(M0'*x)*U0
att=M0'*x
plot(M0*(att .* U0),"--") # =Y1


# Compare stuff in the power domain - reasonable taken into account that we want to get rid of noise
figure()
plot(U0)
plot((spectrogram(reshape(X[:,p],129,1), m; fs=fs, window=w) |> power)[.~ic],"--")

plot(att .* U0)
clamp!(att, 0, 1000000)
btt=zeros(129)
btt[.~ic]=att
plot( (spectrogram(reshape(sqrt.(btt) .* X[:,p],129,1), m; fs=fs, window=w) |> power)[.~ic],"--")



R=similar(X)
for p in 1:size(X,2)
    U0=X2[:,p][.~ic]  # ̃x regular version of this input
    M0=M[:,.~ic]      # ̃M regular version of M
    Y0=M0*U0          # ̃y0 regular version of input
    Y1=Y2[:,p]         # y1 regular version of output
    x=(M0*Diagonal(U0)*M0')\Y1  #use w for weights

    att=M0'*x            #PowerAttenuation
    clamp!(att, 0, 1)
    btt=zeros(129)        # attenuation
    btt[.~ic]=att
    R[:,p]=sqrt.(btt) .* X[:,p]  # Result
end



#R matches Y1 in the power domain
p=45
figure()
#plot(M0*Diagonal(M0'*x)*U0)
Y1=Y2[:,p]
plot(Y1)     # => Y1=M0*Diagonal(M0'*x)*U0
plot(M*(spectrogram(reshape(R[:,p],129,1), m; fs=fs, window=w) |> power),"--")

#!!! NOW CHECK MAP FROM Y2'S WITH MAP FROM R

# Can I regenerate my color map from Y2 alone?

Q=copy(Y2)
Q=pow2db.(Q)
whiten!(Q)

figure(figsize=(10, 5))
imshow(Q, aspect = "auto")
title("white NS out")
colorbar()

# Same process starting from R, seems like it, what was the problem?

RR=spectrogram(R, m; fs=fs, window=w) |> power
R2=M*RR
RW=copy(R2)
RW=pow2db.(RW)
whiten!(RW)
figure(figsize=(10, 5))
imshow(RW, aspect = "auto")


R


y=real.(antistft(R))

sound(y, fs)
sound(t, fs)

####################################################################

plot(M*(spectrogram(reshape(X[:,p],129,1), m; fs=fs, window=w) |> power))





RR=spectrogram(R, m; fs=fs, window=w) |> power

R2=power_chain(X, m, fs, w)|> 
    clip_noise |>
    U -> freq_chain(U, M)

white=whiten!(R2)

figure(figsize=(10, 5))
imshow(R2, aspect = "auto")
title("raw MX2")
colorbar()


#end wrong



figure()
plot(sqrt.(att))

figure()
plot(M*(spectrogram(reshape(X[:,p],129,1), m; fs=fs, window=w) |> power))

figure()
plot(M0'*x)



U0

RR=zeros(ComplexF64,length(ic))
RR[.~ic]=att .* X[.~ic,p]

R2=spectrogram(reshape(RR,129,1), m; fs=fs, window=w) |> power
plot(M*R2)


figure()
plot(Y0)
plot(Y1)


for p in 1:size(X2,2)
    U0=X2[:,p][.~ic]
    M0=M[:,.~ic]

    Y0=M0*U0
    Y1=Y2[:,p]

    x=(M0*Diagonal(U0)*M0')\Y1


    att=M0'*x
    clamp!(att, 0, 1)
    att=sqrt.(att)

    #aa[:,p]=att
    R[.~ic,p]= X[.~ic,p] .* att
end





#bb=zeros(size(X2))
#bb[.~ic,:]=aa
#figure(figsize=(10, 5))
#imshow(pow2db.(X2 .* bb), aspect = "auto")
#colorbar()


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

sound(t, fs)

y=real.(antistft(R))
#y *= sum(abs2.(signals.test[8]+noises.test[4]))/sum(abs2.(y))

sound(y, fs)
wavwrite(y,"result.wav";Fs=8192)

#figure(figsize=(10, 5))
#imshow(aa, aspect = "auto")
#colorbar()




figure(figsize=(10, 5))
imshow(pow2db.(X2), aspect = "auto")
colorbar()


##############

#CONVERSION REMINDERS
abs2.(X[:,p]) ./ X2[:,p]
(spectrogram(reshape(X[:,p],129,1), m; fs=fs, window=w) |> power) ./ X2[:,p]
