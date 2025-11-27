import numpy as np
import soundfile as sf
fs = 44_000
f0 = 80                     # 想要的基频
R  = int(np.round(fs/f0))
g  = 0.9999
x  = np.random.randn(fs*2)   # 2 s 白噪
y  = np.zeros_like(x)
for n in range(R, len(x)):
    y[n] = x[n] + g*y[n-R]
sf.write("breathy_far.wav", y, fs)
print("Saved: breathy_far.wav")