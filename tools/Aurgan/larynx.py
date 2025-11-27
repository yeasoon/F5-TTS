import numpy as np
import soundfile as sf
from scipy.signal import hilbert, butter, sosfilt, lfilter, filtfilt
from scipy.fft import rfft, irfft

def extract_envelope(x, sr, cutoff=8):
    analytic = hilbert(x)
    env = np.abs(analytic)
    sos = butter(2, cutoff, btype="low", fs=sr, output="sos")
    env_smooth = sosfilt(sos, env)
    env_smooth /= np.max(env_smooth) + 1e-9
    return env_smooth

def asymmetric_glottal_wave(phase):
    """Approximation of LF glottal flow derivative."""
    phase = np.mod(phase, 2 * np.pi)
    return np.where(phase < 3*np.pi/2,
                    0.1 * (1 - np.cos(phase)),  # smooth rise
                    -0.2 * np.exp(-3 * (phase - np.pi)))  # sharp closure

def breath_driven_glottal(breath, sr, f0_mean=100, jitter_strength=0.1, drift_strength=1000):
    n = len(breath)
    t = np.linspace(0, n / sr, n)
    envelope = extract_envelope(breath, sr)

    # F0 variation
    drift = np.cumsum(np.random.randn(n)) / n
    drift = 1.0 + drift_strength * (drift - np.mean(drift))*0
    jitter = 1.0 + jitter_strength * np.random.randn(n)*0
    f0_inst = f0_mean * jitter + drift
    print(f0_inst)
    print(np.cumsum(f0_inst))
    # Integrate phase
    print(f0_inst.shape)
    phase = 2 * np.pi * np.cumsum(f0_inst) / sr
    print(phase.shape)
    glottal = asymmetric_glottal_wave(phase)
    glottal *= envelope

    # Gentle lowpass filter to remove harshness
    # sos = butter(2, 6000, btype='low', fs=sr, output='sos')
    # glottal = sosfilt(sos, glottal)
    glottal /= np.max(np.abs(glottal)) + 1e-9
    return glottal

def formant_filter(x, sr, f, bw):
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * f / sr
    b = [1 - r]
    a = [1, -2 * r * np.cos(theta), r ** 2]
    return lfilter(b, a, x)

VOWEL_FORMANTS = {
    "i":  [(270, 60), (2290, 90), (3010, 150)],   # see
    "e":  [(530, 80), (1840, 90), (2480, 150)],   # say
    "ɛ":  [(610, 80), (1900, 100), (2480, 150)],  # bed
    "æ":  [(850, 100), (1610, 120), (2500, 180)], # cat
    "ɑ":  [(730, 80), (1090, 90), (2440, 150)],   # father
    "ɔ":  [(570, 80), (840, 90), (2410, 150)],    # thought
    "u":  [(300, 60), (870, 80), (2240, 120)],    # goose
    # "o":  [(300, 60), (870, 80), (2240, 120)],    # goose
    "ʊ":  [(440, 80), (1020, 100), (2240, 150)],  # foot
    "ə":  [(500, 80), (1500, 100), (2500, 150)],  # sofa
}
def exp_glide(F_start, F_end, n):
    t = np.linspace(0,1,n)
    # exponential curve: fast start, slow end
    F = F_start * (F_end/F_start)**t
    return F

def power_glide(f_start, f_end, n, p=0.15):
    """t in [0,1], smaller p means faster start"""
    t = np.linspace(0,1,n)
    t_mod = t ** p
    return f_start + (f_end - f_start) * t_mod

def gen_change(voice, sr, f0,f1):
    n=len(voice)
    # Formant trajectory: F1, F2, F3 change over time
    F1 = power_glide(f0[0][0], f1[0][0], n)   # /a/ 730 → /o/ 570
    F2 = power_glide(f0[1][0], f1[1][0], n)
    F3 = power_glide(f0[2][0], f1[2][0], n)
    Bs = np.array([f0[0][1],f0[1][0],f0[2][1]])  # can keep bandwidth fixed or vary similarly
    # Bs = np.array([5,5,5])  # can keep bandwidth fixed or vary similarly
    y = np.zeros_like(voice)
    dt = 1/sr
    v1=v2=v3=y1=y2=y3=0.0
    for i in range(n):
        # formant 1
        dv = voice[i] - 2*np.pi*Bs[0]*v1 - (2*np.pi*F1[i])**2*y1
        v1 += dv*dt
        y1 += v1*dt
        # formant 2
        dv = voice[i] - 2*np.pi*Bs[1]*v2 - (2*np.pi*F2[i])**2*y2
        v2 += dv*dt
        y2 += v2*dt
        # formant 3
        dv = voice[i] - 2*np.pi*Bs[2]*v3 - (2*np.pi*F3[i])**2*y3
        v3 += dv*dt
        y3 += v3*dt
        y[i] = y1 + y2 + y3
    y /= np.max(np.abs(y)+1e-9)
    return y

def noise2gottal(noise, sr, harm=30, f0=100, bw=10):
    N = len(noise)
    t = np.linspace(0, N / sr, N)
    # envelope = extract_envelope(breath, sr)
    X = rfft(noise)
    freq = np.fft.rfftfreq(N, 1/sr)
    mask  = np.zeros_like(freq) !=0
    for h in range(1, harm+1):
        f_lo = h*f0 - bw/2
        f_hi = h*f0 + bw/2
        # print(f_lo, f_hi)
        mask = (freq >= f_lo) & (freq <= f_hi) | mask
        if h >2:
            f_lo =f0/h - bw/2
            f_hi = f0/h + bw/2
            # print(f_lo, f_hi)
            mask = (freq >= f_lo) & (freq <= f_hi) | mask
    X[~mask] = X[~mask]*6e-8
    X[mask] = X[mask]*2e10
    y = irfft(X)
    # y *= envelope

    y/= np.max(np.abs(y))
    return y

def noise2gottalv2(noise, sr, f0=100, g=0.999):                   # 想要的基频
    R_mean  = int(np.round(sr/f0))
    # R  = int(np.round(sr/f0))
    y  = np.zeros_like(noise)
    # for n in range(R_mean, len(noise)):
    #     # y[n] = noise[n] + g*y[n-R]
    #     R = int(R_mean + np.random.randint(-9, 10))  # jitter ±2 samples
    #     y[n] = noise[n] + g * y[n - R]
    # for detune in np.linspace(-5, 5, 9):
    #     R = int(np.round(sr / (f0 + detune)))
    #     tmp = np.zeros_like(noise)
    #     for n in range(R, len(noise)):
    #         tmp[n] = noise[n] + g * tmp[n - R]
    #     y += tmp
    # y /= 9
    N = sr * 2
    amp = np.linspace(1, 0.2, N//2+1)     # gently roll off highs
    phase = np.random.rand(N//2+1) * 2*np.pi
    Y = amp * np.exp(1j * phase)
    y = np.fft.irfft(Y)
    y/= np.max(np.abs(y))
    return y

def synthesize_vowel(breath, sr=24000):
    glottal = breath_driven_glottal(breath, sr)
    # glottal = noise2gottal(breath, sr,harm=4000,f0=100, bw=1)
    # glottal = noise2gottalv2(breath, sr,f0=100, g=0.9999)
    # glottal += 5e-3*breath
    # glottal = breath
    # return glottal
    # return glottal

    # /ɑː/ formants
    # F = [730, 1090, 2440]#a
    # F = [500, 1500, 2000] #e
    
    # BW = [80, 90, 120]

    out = glottal
    l=len(out)
    a=int(0.1*l)
    def get_voice(inp, char):
        # for f, bw in zip(F, BW):
        for f, bw in VOWEL_FORMANTS[char]:
            inp = formant_filter(inp, sr, f, bw)
        # out += 1e-4*breath
        inp /= np.max(np.abs(inp)) + 1e-9
        return inp
    char0 = list(VOWEL_FORMANTS.keys())[4]
    out = get_voice(out, char0)
    # char = list(VOWEL_FORMANTS.keys())[5]
    # out= gen_change(out, sr, VOWEL_FORMANTS[char0],VOWEL_FORMANTS[char])


    # out[a:] = get_voice(out[a:], char)
    return out
    
def pink_noise(n):
    out = np.zeros(n)
    for k in range(12):
        step = 2**k
        rand = np.random.randn((n + step - 1)//step)
        out += np.repeat(rand, step)[:n]

    return out / np.max(np.abs(out) + 1e-9)
def fft_pink(n, sr):
    # --- Step 1: white noise ---
    x = np.random.randn(n)

    # --- Step 2: smooth it in time to avoid harsh edges ---
    def smooth_noise(x, cutoff=1000, sr=22050, order=4):
        b, a = butter(order, cutoff / (sr / 2), btype='low')
        return filtfilt(b, a, x)

    # x = smooth_noise(x, cutoff=10000, sr=sr)

    # --- Step 3: Pink shaping (1/sqrt(f)), skip f=0 ---
    f = np.fft.rfftfreq(n, 1 / sr)
    spectrum = np.fft.rfft(x)
    print(spectrum.shape)

    # avoid f=0 division issue
    pink_filter = np.ones_like(f)
    # pink_filter[1:] = 1 / np.sqrt(f[1:])  # only apply to nonzero freqs
    pink_filter[1:] = 1 / np.log(f[1:]+1)  # only apply to nonzero freqs

    spectrum *= pink_filter
    # print(np.min(spectrum), np.max(spectrum), np.mean(spectrum))
    # spectrum = np.max(spectrum)*spectrum/spectrum
    y = np.fft.irfft(spectrum)

    # normalize
    y /= np.max(np.abs(y))
    return y
def pink_voss(n, n_rows=16):
    """Voss–McCartney algorithm for smooth pink noise"""
    array = np.random.randn(n_rows, n)
    array = np.cumsum(array, axis=1)
    weights = 2.0 ** np.arange(-n_rows, 0)
    y = np.dot(weights, array)
    return y / np.max(np.abs(y))




def pink_iir(n, sr=22050):
    x = np.random.randn(n)
    def pink_filter_iir(x):
        """
        Pink noise filter (IIR approximation by Paul Kellet).
        Keeps high frequencies and ensures smooth output.
        """
        b = [0.049922035, 0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        y = lfilter(b, a, x)
        return y
    y = pink_filter_iir(x)
    y /= np.max(np.abs(y))
    return y
def fractional_brownian(n, H=0.7):
    """Generate fractional Brownian noise (H=0.5 white, 1.0 Brown)."""
    white = np.random.randn(n)
    y = np.cumsum(white)
    y = np.diff(np.convolve(y, np.ones(8)/8, mode='same'))
    y /= np.max(np.abs(y))
    return y
if __name__ == "__main__":
    sr = 24000
    t = np.linspace(0, 2.0, int(sr * 2.0))
    n = len(t)
    # breath = np.random.randn(len(t)) * np.exp(-3 * t / 2.0)
    breath = pink_noise(n) * np.exp(-3 * t / 2.0)
    # breath = fft_pink(n, sr) * np.exp(-3 * t / 2.0)
    # breath = pink_iir(n, sr) * np.exp(-3 * t / 2.0)
    
    # sos = butter(2, 8000, btype='low', fs=sr, output='sos')
    # breath = sosfilt(sos, breath)
    breath /= np.max(np.abs(breath))

    vowel = synthesize_vowel(breath, sr)
    # sf.write("vowel_aa_fixed.wav", vowel, sr)
    # print("✅ Saved: vowel_aa_fixed.wav (more natural /ɑː/ with LF-like glottal shape)")

    sf.write("breathy_far.wav", vowel, sr)
    print("Saved: breathy_far.wav")


# import numpy as np
# import soundfile as sf
# from scipy.signal import butter, sosfilt

# sr = 22050
# duration = 0.5
# n = int(sr*duration)
# t = np.linspace(0, duration, n)

# # Generate breath noise
# noise = np.random.randn(n)

# # Formant trajectory: F1, F2, F3 change over time
# F1 = np.linspace(730, 570, n)   # /a/ 730 → /o/ 570
# F2 = np.linspace(1090, 840, n)
# F3 = np.linspace(2440, 2410, n)
# BW = np.array([80,90,150])  # can keep bandwidth fixed or vary similarly

# # Frame size
# frame_len = 512
# hop = 256
# output = np.zeros_like(noise)

# def time_varying_formant(frame, sr, f, bw):
#     y = np.zeros_like(frame)
#     for fi, bwi in zip(f,bw):
#         low = (fi - bwi/2)/(sr/2)
#         high = (fi + bwi/2)/(sr/2)
#         sos = butter(2,[low,high],btype='band',output='sos')
#         y += sosfilt(sos, frame)
#     y /= np.max(np.abs(y)+1e-9)
#     return y

# # Process frame by frame
# for start in range(0, n-frame_len, hop):
#     end = start+frame_len
#     frame = noise[start:end]
#     f_frame = [F1[start:end].mean(), F2[start:end].mean(), F3[start:end].mean()]
#     y_frame = time_varying_formant(frame, sr, f_frame, BW)
#     output[start:end] += y_frame

# output /= np.max(np.abs(output)+1e-9)
# sf.write("breathy_far.wav", output, sr)
# print("✅ Saved: breathy_far.wav")
