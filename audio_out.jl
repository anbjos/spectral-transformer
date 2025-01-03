signals
noises

m=256
w=hanning(m+1)[2:m+1]
M=mel_filter_bank(26, m, fs, 300, fs/2)

#R=zeros(ComplexF64,size(X))

t=signals.test[5]+noises.test[7]
x=t
X=stft(x[:], m; fs=fs, window=w)
XX=spectrogram(X, m; fs=fs, window=w) |> power

X2=power_chain(X, m, fs, w)|> 
    clip_noise |>
    U -> freq_chain(U, M)

XW=copy(X2)
white=whiten!(XW)

figure(figsize=(10, 5))
imshow(XW, aspect = "auto")
title("raw XW")
colorbar()

sample=(x=reshape(XW,26,127,1) |> todevice,y=reshape(XW,26,127,1) |> todevice,mask=[size(XW,2)] |> todevice) |> todevice


YW=model(withmask(sample).x ).hidden_state |> cpu
YW=reshape(YW,size(YW,1),size(YW,2))

#XX=spectrogram(R, m; fs=fs, window=w) NOTE - the result is terrible compared to Y2!!!!!!!!

figure(figsize=(10, 5))
imshow(YW, aspect = "auto")
title("clean YW")
colorbar()

function antiwhiten!(x,white)
    x.*=white.scale
    x.+=white.offset
    return x
end

Y2=copy(YW)
antiwhiten!(Y2,white)

Y2=db2pow.(Y2) #Preserves the information from YW.

sum(Y2)/sum(db2pow.(X2))

#Y2 is power in mel domain

#Compare before and after NS in MEL domain
p=45
clf()
plot(Y2[:,p])
plot(db2pow.(X2[:,p]))


ignored_columns(W)=[iszero(c) for c in eachcol(W)] |> BitVector
ic=ignored_columns(M)



p=45
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


#Still something is wrong....
#Try to compare results in power domain....!!!!!!!!!!!!!!!
R=similar(X)
for p in 1:size(X,2)
    U0=XX[:,p][.~ic]
    M0=M[:,.~ic]
    Y0=M0*U0
    Y1=Y2[:,p]
    x=(M0*Diagonal(U0)*M0')\Y1

    att=M0'*x
    clamp!(att, 0, 1)
    btt=zeros(129)
    btt[.~ic]=att
    R[:,p]=sqrt.(btt) .* X[:,p]
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


plot(M*(spectrogram(reshape(X[:,p],129,1), m; fs=fs, window=w) |> power))





RR=spectrogram(R, m; fs=fs, window=w) |> power

R2=power_chain(X, m, fs, w)|> 
    clip_noise |>
    U -> freq_chain(U, M)

white=whiten!(R2)

figure(figsize=(10, 5))
imshow(R2, aspect = "auto")
title("raw X2")
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


for p in 1:size(XX,2)
    U0=XX[:,p][.~ic]
    M0=M[:,.~ic]

    Y0=M0*U0
    Y1=Y2[:,p]

    x=(M0*Diagonal(U0)*M0')\Y1


    att=M0'*x
    clamp!(att, 0, 1)
    att=sqrt.(att)

    aa[:,p]=att
    R[.~ic,p]= X[.~ic,p] .* att
end





bb=zeros(size(XX))
bb[.~ic,:]=aa
figure(figsize=(10, 5))
imshow(pow2db.(XX .* bb), aspect = "auto")
colorbar()


















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
endy=real.(antistft(R))

sound(t, fs)

y *= sum(abs2.(signals.test[8]+noises.test[4]))/sum(abs2.(y))

sound(y, fs)

figure(figsize=(10, 5))
imshow(aa, aspect = "auto")
colorbar()

aa


figure(figsize=(10, 5))
imshow(pow2db.(XX), aspect = "auto")
colorbar()


##############

#CONVERSION REMINDERS
abs2.(X[:,p]) ./ XX[:,p]
(spectrogram(reshape(X[:,p],129,1), m; fs=fs, window=w) |> power) ./ XX[:,p]
