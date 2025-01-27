# Activate environment

using Pkg
Pkg.activate(".")

# Check that we can play a sound

using Sound

fs=8192
t=(0:fs<<1-1)/fs
f=440
u = sin.(2π*f*t) # A440 tone for 2 seconds
sound(u, fs)

# Read a wav file and play it

using WAV

u, fs = wavread("./data/environmental_sound_classification_50/archive/audio/audio/16000/1-17970-A-4.wav")
sound(u, fs)

# Decimate to fs=8192 Hz

using DSP

u=resample(u,8192/fs;dims=1)
fs=8192
sound(u, fs)

# Alignment for tweaked Hanning window with a step size of n÷2

using Test

fs=8192
n=256
w=hanning(n+1)[2:n+1]
@test w[n>>1]==1
@test w[n>>1+n>>1]==0

# Check that full circle work, and also that the negative halv is simply dismissed in onesided (-and not corrected for)

fullcircle(X)=vcat(X,conj.(X[end-1:-1:2]))

stimuli=stft(u[:], n; fs=fs, window=w)
expected=stft(u[:], n; onesided=false, fs=fs, window=w)

@test all(fullcircle(stimuli[:,5]) .== expected[:,5] )

# Construct STFT and invert back to time domail, carefully aligning STFT using that sin(u)^2+cos(u)^2=1

using FFTW

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

UT=stft(u[:], n; fs=fs, window=w)
y=real.(antistft(UT))

@test isapprox(y[n>>1+1:end-n>>1], u[n>>1+1:end-n>>1])

# Mel filter bank stuff

using PyPlot
pygui(true)

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

A=mel_filter_bank(10, 252, 8000, 300, 4000)
[plot(r) for r in eachrow(A)] # compare with http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/


title("MEL filter bank")
ylabel("weight")
xlabel("stft bin")
legend(["row $k" for k in 1:10];loc="lower right")

# Investigate from DSP's stft to spectrogram

PT=spectrogram(u[:], n; fs=fs, window=w)
UT=stft(u[:], n; fs=fs, window=w)

P=power(PT)[:,1]
U=UT[:,1]


@test sum(isapprox.(abs2.(U)/fs/sum(abs2.(w)),P))==2
@test sum(isapprox.(2abs2.(U)/fs/sum(abs2.(w)) , P))==127
@test 2+127==length(P)

# Expand DSP spectogram to take a stft argument

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

UT=stft(u[:], 256; fs=fs, window=w)
expected=spectrogram(u[:], 256; fs=fs, window=w)
result=spectrogram(UT, 256; fs=fs, window=w)

@test isapprox(power(expected), power(result); rtol=1e-3, atol=1e-6)

# When we de a MEL transform and transform back using the transpose MEL matrix, we are essentially doing a smooting

using LinearAlgebra

PT=spectrogram(u[:], n; fs=fs, window=w)

p=10
U=power(PT)[:,p]

num_filters=26
n 
fs
min_freq=300
max_freq=8192>>1

M=mel_filter_bank(num_filters, n, fs, min_freq, max_freq)


figure()
plot(U)

ignored_columns(W)=[iszero(c) for c in eachcol(W)] |> BitVector
ic=ignored_columns(M)


Y0=M*U
V=zeros(size(U))
g=sum(M)/length(Y0)


V[.~ic]=M[:,.~ic]'*Y0/g
plot(V,"--")

# I do not want to smooth the specturm, but to calculate a attention and apply this to the spectrum
# Idea, use M'*x to define attenuation, thereby resulting underdetermined issue

ignored_columns(W)=[iszero(c) for c in eachcol(W)] |> BitVector
ic=ignored_columns(M)

U0=U[.~ic]
M0=M[:,.~ic]

sum(U)

Y0=M0*U0

x=(M0*Diagonal(U0)*M0')\Y0


plot(M0'*x)

V=M'
V*Y0

using LinearAlgebra
plot(sum(V;dims=2))

# A more practical inplementaiton of the same

Y0=M[:,.~ic]*U[.~ic]
plot(Y0)
Y1=copy(Y0)
Y1[14:16]=0.01*Y1[14:16]

x=(M[:,.~ic]*Diagonal(U[.~ic])*M[:,.~ic]')\Y1 .|> x -> clamp(x,0.0,1.0)
plot(x)

attenuation=M[:,.~ic]'*x
plot(attenuation)

V=zeros(size(U))
V[.~ic]=attenuation .* U[.~ic]

plot(U)
plot(V,"--")

# Another optios is to limit each bin so that it could not violate Y1, even if all other bins was 0. 
# This is more relaxed, but solve the ringing issue that may exist at the edges.


function posenc(i,j,d=32)
    if iseven(i)
        result=sin(j/10000^(i/d))
    else
        result=cos(j/10000^((i-1)/d))
    end
end

pe=[posenc(i,j) for i in 0:31, j in 0:15]

figure(figsize=(10, 5))
imshow(pe, aspect = "auto",cmap="Blues")
title("positional encoding\n d=32, n=16")
colorbar()
ylabel("Position Embedding")
xlabel("Sequence")

pe[:,1]' * pe[:,5]

d=32

cc(j,k)=[cos((j-k)/1000^(i/d)) for i in 0:d-1]
sum(cc(1,5))

c=[pe[:,i]' * pe[:,j] for i in 1:16, j in 1:16]
figure(figsize=(10, 5))
imshow(c, aspect = "auto", cmap="Blues")
title("Position Embedding relative distance\n n=16")
xlabel("Sequence")
ylabel("Sequence")
colorbar()

print(plt.colormaps())
PyPlot.cm

# Audio reconstruction

# Signals and Noises

signals
noises

# Define various parameters

m=256
w=hanning(m+1)[2:m+1]
M=mel_filter_bank(26, m, fs, 300, fs/2)

# Define and process time domain signal

t=signals.test[2]+noises.test[9]
x=t
X=stft(x[:], m; fs=fs, window=w)
XX=spectrogram(X, m; fs=fs, window=w) |> power
X2=power_chain(X, m, fs, w)|> clip_noise |> U -> freq_chain(U, M)

whiteX2=copy(X2)
white=whiten!(whiteX2)

figure(figsize=(10, 5))
imshow(whiteX2, aspect = "auto")
title("raw whiteX2")
colorbar()

sample=(x=reshape(whiteX2,26,127,1) |> todevice,y=reshape(whiteX2,26,127,1) |> todevice,mask=[size(whiteX2,2)] |> todevice) |> todevice


whiteY2=model(withmask(sample).x ).hidden_state |> cpu
whiteY2=reshape(whiteY2,size(whiteY2,1),size(whiteY2,2))

figure(figsize=(10, 5))
imshow(whiteY2, aspect = "auto")
title("clean whiteY2")
colorbar()

function antiwhiten!(x,white)
    x.*=white.scale
    x.+=white.offset
    return x
end

Y2=copy(whiteY2)
antiwhiten!(Y2,white)

Y2=db2pow.(Y2) #Preserves the information from whiteY2.

println("output to input power ratio: $(sum(Y2)/sum(db2pow.(X2)))")

# select a sftf and compare input and output power spectrum
p=45
clf()
plot(Y2[:,p])
plot(db2pow.(X2[:,p]))
title("Linear Power ")
xlabel("Frequency, MEL logarithmic")
ylabel("Power")

ignored_columns(W)=[iszero(c) for c in eachcol(W)] |> BitVector
ic=ignored_columns(M)

U0=XX[:,p][.~ic]
M0=M[:,.~ic]
Y0=M0*U0
Y1=Y2[:,p]

x=(M0*Diagonal(U0)*M0')\Y1

#Verify match att*XX to Y2 in MEL domain. THIS MATCHES
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
    U0=XX[:,p][.~ic]  # ̃x regular version of this input
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

y=real.(antistft(R))

sound(y, fs)
sound(t, fs)

#################
# Analysis of results 

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
