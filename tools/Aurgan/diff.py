import numpy as np
import soundfile as sf

sr = 22050

# ----------------------------
# 1. Second-order differential bandpass filter (discrete)
# ----------------------------
def differential_formant(Fn, Bn, source, dt):
    """
    Fn: center frequency (Hz)
    Bn: bandwidth (Hz)
    source: excitation signal
    dt: 1/sr
    """
    omega = 2 * np.pi * Fn
    alpha = np.pi * Bn
    y = np.zeros_like(source)
    v = 0.0
    for n in range(len(source)):
        # Differential update
        dv = source[n] - 2*alpha*v - omega**2*y[n-1] if n>0 else source[n] - 2*alpha*v
        v += dv*dt
        y[n] = y[n-1]+v*dt if n>0 else v*dt
    # Normalize
    y /= np.max(np.abs(y)+1e-9)
    return y

# ----------------------------
# 2. Breath source generator
# ----------------------------
def breath_source(duration, envelope=None):
    n = int(sr*duration)
    noise = np.random.randn(n)
    if envelope is None:
        # default: smooth rise and fall
        envelope = np.linspace(0,1,n//2)
        envelope = np.concatenate([envelope, envelope[::-1]])
        if len(envelope) < n:
            envelope = np.pad(envelope,(0,n-len(envelope)))
    return noise * envelope

# ----------------------------
# 3. Multi-formant vowel synthesis
# ----------------------------
def breath_vowel(Fs, Bs, duration=0.5):
    dt = 1/sr
    src = breath_source(duration)
    y = np.zeros_like(src)
    for f,b in zip(Fs,Bs):
        y += differential_formant(f,b,src,dt)
    y /= np.max(np.abs(y)+1e-9)
    return y

# ----------------------------
# 4. Breath-driven plosive /p/
# ----------------------------
def breath_p(duration=0.03):
    burst = np.random.randn(int(sr*duration))
    burst *= np.hanning(len(burst))
    burst /= np.max(np.abs(burst)+1e-9)
    return burst

# ----------------------------
# 5. Dynamic vowel trajectory: /a/ -> /o/
# ----------------------------
def breath_diphthong(F_start,B_start,F_end,B_end,duration=0.5):
    n = int(sr*duration)
    # src = breath_source(duration)
    y = np.zeros_like(src)
    dt = 1/sr
    F1 = np.linspace(F_start[0],F_end[0],n)
    F2 = np.linspace(F_start[1],F_end[1],n)
    F3 = np.linspace(F_start[2],F_end[2],n)
    Bs = B_start
    # simulate sample by sample for time-varying F
    v1=v2=v3=y1=y2=y3=0.0
    for i in range(n):
        # formant 1
        dv = src[i] - 2*np.pi*Bs[0]*v1 - (2*np.pi*F1[i])**2*y1
        v1 += dv*dt
        y1 += v1*dt
        # formant 2
        dv = src[i] - 2*np.pi*Bs[1]*v2 - (2*np.pi*F2[i])**2*y2
        v2 += dv*dt
        y2 += v2*dt
        # formant 3
        dv = src[i] - 2*np.pi*Bs[2]*v3 - (2*np.pi*F3[i])**2*y3
        v3 += dv*dt
        y3 += v3*dt
        y[i] = y1 + y2 + y3
    y /= np.max(np.abs(y)+1e-9)
    return y

# ----------------------------
# 6. Generate "tao" syllable fully breath-driven
# ----------------------------
# /p/ burst
# p_burst = breath_p(0.03)

# /a/ -> /o/ diphthong
Fs_start = [730,1090,2440]  # /a/
Fs_end   = [570,840,2410]   # /o/
Bs = [80,90,150]

diph = breath_diphthong(Fs_start,Bs,Fs_end,Bs,duration=0.5)

# Combine
# tao = np.concatenate([p_burst,diph])
tao=diph
tao /= np.max(np.abs(tao)+1e-9)

# ----------------------------
# 7. Save
# ----------------------------
sf.write("breathy_far.wav",tao,sr)
print("✅ Saved: breathy_far.wav")
