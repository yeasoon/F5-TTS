import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sr = 22050  # sample rate

# ----------------------------
# 1. Bandpass filter for formants / fricatives
# ----------------------------
def bandpass_filter(sig, sr, center, bandwidth):
    low = (center - bandwidth/2) / (sr/2)
    high = (center + bandwidth/2) / (sr/2)
    sos = butter(2, [low, high], btype='band', output='sos')
    return sosfilt(sos, sig)

# ----------------------------
# 2. Breath-driven vowel generator
# ----------------------------
VOWEL_FORMANTS = {
    "ɑ": [(730,80),(1090,90),(2440,150)],
    "i": [(270,60),(2290,90),(3010,150)],
    "u": [(300,60),(870,80),(2240,120)],
}

def breath_vowel(vowel, duration=0.5):
    noise = np.random.randn(int(sr*duration))
    formants = VOWEL_FORMANTS[vowel]
    y = np.zeros_like(noise)
    for f,bw in formants:
        y += bandpass_filter(noise, sr, f, bw)
    y /= np.max(np.abs(y)+1e-9)
    return y

# ----------------------------
# 3. Breath-driven plosive /p, t, k/
# ----------------------------
PLS_FRQ = {
    "p": (500, 4000),
    "t": (1000, 6000),
    "k": (800, 5000),
}

def breath_plosive(c, burst_duration=0.03):
    burst = np.random.randn(int(sr*burst_duration))
    burst *= np.hanning(len(burst))
    # bandpass
    low, high = PLS_FRQ[c]
    freqs = np.fft.rfftfreq(len(burst),1/sr)
    fft = np.fft.rfft(burst)
    mask = (freqs < low) | (freqs > high)
    fft[mask] *= 0.2
    burst = np.fft.irfft(fft)
    return burst/np.max(np.abs(burst)+1e-9)

# ----------------------------
# 4. Breath-driven fricative /s, f, h/
# ----------------------------
FRC_FRQ = {
    "s": (4000,8000),
    "f": (2000,5000),
    "h": (300,6000),
}

def breath_fricative(c, duration=0.1):
    noise = np.random.randn(int(sr*duration))
    low, high = FRC_FRQ[c]
    freqs = np.fft.rfftfreq(len(noise),1/sr)
    fft = np.fft.rfft(noise)
    mask = (freqs < low) | (freqs > high)
    fft[mask] *= 0.2
    y = np.fft.irfft(fft)
    return y/np.max(np.abs(y)+1e-9)

# ----------------------------
# 5. CV syllable generator
# ----------------------------
def breath_cv(c,v):
    if c in PLS_FRQ:
        cons = breath_plosive(c)
    elif c in FRC_FRQ:
        cons = breath_fricative(c)
    else:
        cons = np.zeros(int(sr*0.05))  # silence for unknown
    vow = breath_vowel(v)
    syllable = np.concatenate([cons, vow])
    syllable /= np.max(np.abs(syllable)+1e-9)
    return syllable

# ----------------------------
# 6. Generate sequence: pa, ta, ka, sa
# ----------------------------
sequence = [("p","ɑ"),("t","ɑ"),("k","ɑ"),("s","ɑ")]
output = np.concatenate([breath_cv(c,v) for c,v in sequence])
output /= np.max(np.abs(output)+1e-9)

# ----------------------------
# 7. Save
# ----------------------------
# sf.write("breath_speech.wav", output, sr)
# print("✅ Saved: breath_speech.wav")

sf.write("p_sound.wav", output, sr)
print("✅ Saved: p_sound.wav")
