import numpy as np
from scipy.signal import lfilter, lfilter_zi, tf2zpk
import matplotlib.pyplot as plt
c=343
Nx=[1,2]
Ny=[0, 1,2]
Nz=[0, 1,2]
L=0.5
W=0.1 
H=0.2
# fxyz =c/2*np.sqrt((Nx/L)**2+(Ny/W)**2+(Nz/H)**2)
# fxyz=[c/2*np.sqrt((Nx/L)**2 for i in Nx]
fxyz=[]
Bxyz=[]
for i in Nx:
    for j in Ny:
        for k in Nz:
            fxyz.append(c/2*np.sqrt((i/L)**2+(j/W)**2+(k/H)**2))
            Bxyz.append(100)

F=fxyz[:10]
idx = sorted(range(len(F)), key=lambda i: F[i])
print(F)
BW=Bxyz[:10]
BW=[BW[i] for i in idx]
fs = 32000  # sampling rate
F = [750, 1200, 2600]  # formants
# BW = [100, 100, 100]    # bandwidths
BW = [80, 100, 150]    # bandwidths
# Compute poles for each formant
a_total = [1.0]
for f, bw in zip(F, BW):
    r = np.exp(-np.pi*bw/fs)
    theta = 2*np.pi*f/fs
    a = [1, -2*r*np.cos(theta), r**2]
    a_total = np.convolve(a_total, a)

print(a_total.shape)
# Impulse response
impulse = np.zeros(1024)
impulse[0] = 1
h = lfilter([1], a_total, impulse)
# print(h.shape)
# Plot frequency response
H = np.fft.fft(h, 2048)
# print(H.shape)
freq = np.fft.fftfreq(2048, 1/fs)


x=[]
for i  in range (28):
    
    d = 2 * (i / 28)
    if d < 1:
        diameter = 0.4 + 1.6 * d
    else:
        diameter = 0.5 + 1.5 * (2 - d)
    diameter = min(diameter, 1.9)
    x.append(diameter)
x=[]
for i  in range (44):
    if (i < 7 *  44/ 44 - 0.5):
        diameter = 0.6
    elif (i < 12 * 44 / 44):
        diameter = 1.1
    else:
        diameter = 1.5
    x.append(diameter)

plt.figure(figsize=(20,20))
# plt.plot(freq[:1024], 20*np.log10(np.abs(H[:1024])))
plt.bar(range(44), x)
plt.title("Frequency Response of vowel 'a'")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.savefig("fir.png")
