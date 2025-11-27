import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, hilbert, lfilter
import random

# -------------------------------------------------
# Helper: filters and noise
# -------------------------------------------------
def bandpass_sos(low, high, sr, order=4):
    return butter(order, [low, high], btype='band', fs=sr, output='sos')

def pink_noise(n):
    out = np.zeros(n)
    for k in range(12):
        step = 2**k
        rand = np.random.randn((n + step - 1)//step)
        out += np.repeat(rand, step)[:n]
    return out / np.max(np.abs(out) + 1e-9)

# -------------------------------------------------
# Breath generation
# -------------------------------------------------
def breath_envelope(length, decay, peak=1.0):
    a = int((1-decay) * length)
    d = int(decay * length)
    env = np.zeros(length)
    # print(length, a, d)
    env[:a] = np.sin(np.linspace(0, np.pi/2, a))**2
    env[a:a+d] = np.exp(-np.linspace(0, 3, d))
    env /= np.max(env)
    return peak * env

def generate_breath(sr, duration, kind="exhale", pink=True):
    n = int(duration * sr)
    noise = pink_noise(n) if pink else np.random.randn(n)
    noise /= np.max(np.abs(noise)) + 1e-9
    sos = bandpass_sos(100, 10000, sr)
    y = sosfilt(sos, noise)

    if kind == "inhale":
        sos = bandpass_sos(1000, 5000, sr)
        y = sosfilt(sos, y)
        env = breath_envelope(n, decay=0.2, peak=1e-10)
    else:
        env = breath_envelope(n, decay=0.2, peak=1.0)
    y *= env
    y /= np.max(np.abs(y)) + 1e-9
    return y


def bandpass_sos(center, bandwidth, sr, order=4):
    low = max(50, center - bandwidth / 2)
    high = min(sr/2 - 100, center + bandwidth / 2)
    return butter(order, [low, high], btype='band', fs=sr, output='sos')

def shape_breath_to_vowel(breath, sr, vowel="far"):
    """
    Shape breath noise into the 'void' sound of a vowel like /ɑː/ (far)
    """
    vowel_formants = {
        "far": [(750, 100, 1.0), (1100, 150, 0.8), (2500, 200, 0.5)],
        "fee": [(300, 60, 1.0), (2300, 150, 0.8), (3000, 200, 0.5)],
        "foo": [(400, 70, 1.0), (900, 100, 0.7), (2500, 200, 0.4)],
        "fur": [(500, 80, 1.0), (1200, 150, 0.8), (2600, 200, 0.5)],
    }
    rand_freq_shift_range={
        "far": [[-5,5], [-5, 5], [-5, 5]],
    }
    rand_freq_shift={}
    for item in rand_freq_shift_range:
        rand_freq_shift[item]=[]
        for range_ in rand_freq_shift_range[item]:
            rand_freq_shift[item].append(random.uniform(*tuple(range_)))
    # rand_freq_shift=
    formants = vowel_formants.get(vowel, vowel_formants["far"])
    shaped = np.zeros_like(breath)
    for idx, x in enumerate(formants):
        f, bw, g =x
        f+=rand_freq_shift[vowel][idx]
        sos = bandpass_sos(f, bw, sr)
        shaped += g * sosfilt(sos, breath)
    shaped /= np.max(np.abs(shaped)) + 1e-9
    return shaped

def normalize(x):
    mx = np.max(np.abs(x)) + 1e-12
    return x / mx

# Rosenberg (or cosine) shaped glottal pulse generator
def rosenberg_pulse(length_samples):
    # Rosenberg: ramp up with half-sine, ramp down with half-sine^2 like shape
    n = length_samples
    t = np.arange(n) / n
    # attack portion (0 -> 0.5)
    attack_len = int(0.3 * n)
    decay_len = n - attack_len
    a = np.sin(np.pi * np.linspace(0, 0.5, attack_len))  # smooth attack
    d = np.sin(np.pi * np.linspace(0.5, 1.0, decay_len)) ** 2  # softer decay
    pulse = np.concatenate([a, d])
    return normalize(pulse)

def generate_glottal_train(sr, duration, f0=120):
    n = int(sr * duration)
    t = np.arange(n) / sr
    # create impulse train at F0 positions
    period = int(sr / f0)
    train = np.zeros(n)
    for i in range(0, n, period):
        train[i] = 1.0
    # convolve with Rosenberg pulse (one glottal open period)
    pulse_len = max(3, int(0.4 * period))  # pulse length ~ 0.3-0.5 of period
    pulse = rosenberg_pulse(pulse_len)
    glot = np.convolve(train, pulse, mode='same')
    # apply mild spectral tilt to reduce high harmonics (1/f^alpha)
    freqs = np.fft.rfftfreq(n, 1/sr)
    G = np.fft.rfft(glot)
    alpha = 0.7  # spectral tilt; increase to mellow more
    tilt = 1.0 / (1.0 + (freqs / (f0*10))**alpha)  # smooth tilt
    G *= tilt
    glot = np.fft.irfft(G, n)
    return normalize(glot)

def generate_glottal_pulse(sr, duration, f0):
    t = np.linspace(0, duration, int(sr * duration))
    
    def get_base(f):
        # Smooth periodic source
        glottal = np.sin(2 * np.pi * f * t)
        glottal = np.sign(glottal) * (np.abs(glottal) ** 3)  # soft waveform
        # Modulate amplitude to feel natural
        # mod = 1 + 0.15 * np.random.randn(len(t))
        # glottal *= np.clip(mod, 0.6, 1.4)

        return glottal / (np.max(np.abs(glottal)) + 1e-9)

    ret= np.sin(2 * np.pi * 0 * t)
    for f,r in f0:
        ret += get_base(f)*r
    return ret / (np.max(np.abs(ret)) + 1e-9)

def extract_envelope(x, sr, cutoff=8):
    """Get smooth energy envelope from noise."""
    analytic = hilbert(x)
    env = np.abs(analytic)
    sos = butter(2, cutoff, btype="low", fs=sr, output="sos")
    env_smooth = sosfilt(sos, env)
    env_smooth /= np.max(env_smooth) + 1e-9
    return env_smooth
def asymmetric_glottal_wave(phase):
    """Approximation of LF glottal flow derivative."""
    phase = np.mod(phase, 2 * np.pi)
    return np.where(phase < np.pi,
                    0.1 * (1 - np.cos(phase)),  # smooth rise
                    -0.8 * np.exp(-3 * (phase - np.pi)))  # sharp closure


def breath_driven_glottal(breath, sr, f0,jitter_strength=0.001, drift_strength=0.002):
    """Generate a glottal-like excitation guided by breath."""
    t = np.linspace(0, len(breath)/sr, len(breath))
    n = len(breath)
    envelope = extract_envelope(breath, sr)
    def get_base(f):
        # Random micro variation of F0 (naturalness)
        f_inst = f * (1 + 0.02 * np.random.randn(len(t)))
        phase = 2 * np.pi * np.cumsum(f_inst) / sr
        
        # Smooth, soft glottal-like waveform
        glottal = np.sin(phase)
        glottal = np.sign(glottal) * (np.abs(glottal) ** 3)
        # glottal=generate_glottal_train(sr, len(breath)/sr, f)
        
        # Apply breath-driven amplitude
        glottal *= envelope
        
        # Normalize
        return glottal / np.max(np.abs(glottal)) + 1e-9

    def get_basev2(f):
        # === F0 dynamics ===
        # Slow drift (like natural speaking intonation)
        drift = np.cumsum(np.random.randn(n)) / n
        drift = (drift - drift.min()) / (drift.max() - drift.min())
        drift = 1.0 + drift_strength * (drift - 0.5)

        # Fast random jitter
        jitter = 1.0 + jitter_strength * np.random.randn(n)

        f0_inst = f * drift * jitter
        phase = 2 * np.pi * np.cumsum(f0_inst) / sr

        # === Glottal pulse shape ===
        # softer and more vocal-like than sine
        glottal = np.sin(phase)
        glottal = np.sign(glottal) * (np.abs(glottal) ** 2.5)

        # === Apply amplitude modulation ===
        glottal *= envelope

        # === Slight spectral shaping (low-pass to remove harsh highs) ===
        sos = butter(2, 6000, btype="low", fs=sr, output="sos")
        glottal = sosfilt(sos, glottal)

        # === Normalize ===
        glottal /= np.max(np.abs(glottal)) + 1e-9
        return glottal
    def get_basev3(f):
        # F0 variation
        drift = np.cumsum(np.random.randn(n)) / n
        drift = 1.0 + drift_strength * (drift - np.mean(drift))
        jitter = 1.0 + jitter_strength * np.random.randn(n)
        f0_inst = f * drift * jitter
        # f0_inst = f +  np.random.randn(n)*0

        # Integrate phase
        phase = 2 * np.pi * np.cumsum(f0_inst) / sr
        glottal = asymmetric_glottal_wave(phase)
        glottal *= envelope

        # Gentle lowpass filter to remove harshness
        sos = butter(2, 6000, btype='low', fs=sr, output='sos')
        glottal = sosfilt(sos, glottal)
        glottal /= np.max(np.abs(glottal)) + 1e-9
        return glottal

    ret= np.sin(2 * np.pi * 0 * t)
    for f,r in f0:
        # ret += get_base(f)*r
        ret += get_basev3(f)*r
    return ret / (np.max(np.abs(ret)) + 1e-9)

def formant_filter(x, sr, f, bw):
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * f / sr
    b = [1 - r]
    a = [1, -2 * r * np.cos(theta), r ** 2]
    return lfilter(b, a, x)

def synthesize_vowel(breath, sr=24000):
    glottal = breath_driven_glottal(breath, sr,[(130,1)])

    # /ɑː/ formants
    F = [730, 1090, 2440]
    BW = [80, 90, 120]

    out = glottal
    for f, bw in zip(F, BW):
        out = formant_filter(out, sr, f, bw)

    out /= np.max(np.abs(out)) + 1e-9
    return out
    
def shape_vowel(breath, glottal, sr, vowel="far"):
    vowel_formants = {
        "far": [(750, 100, 1.0), (1100, 150, 0.8), (2500, 200, 0.5)],
        "foo": [(400, 70, 1.0), (900, 100, 0.7), (2500, 200, 0.4)],
        "fee": [(300, 60, 1.0), (2300, 150, 0.8), (3000, 200, 0.5)],
    }

    fset = vowel_formants[vowel]
    source = 0.3 * breath
    shaped = np.zeros_like(source)
    for f, bw, g in fset:
        sos = bandpass_sos(f, bw, sr)
        shaped += g * sosfilt(sos, source)
        shaped += 0.7 * glottal
    shaped /= np.max(np.abs(shaped)) + 1e-9
    return shaped
# -------------------------------------------------
# Breathing loop generator
# -------------------------------------------------
def generate_breathing_loop(
    duration_total=10.0,
    sr=24000,
    inhale_range=(0.5, 1.0),
    exhale_range=(6.0, 8.0),
    pause_range=(0.01, 0.02),
    pink=True
):
    t = 0.0
    samples = []

    while t < duration_total:
        # inhale
        print(t)
        inh_dur = random.uniform(*inhale_range)
        inhale = generate_breath(sr, inh_dur, "inhale", pink)
        samples.append(inhale)
        t += inh_dur

        # pause
        pause1 = np.zeros(int(random.uniform(*pause_range) * sr))
        samples.append(pause1)
        t += len(pause1)/sr

        # exhale
        ex_dur = random.uniform(*exhale_range)
        exhale = generate_breath(sr, ex_dur, "exhale", pink)
        # shape_breath_to_vowel()
        # vowel="far"
        # vowel="fee"
        # vowel="foo"
        # vowel="fur"
        # exhale=shape_breath_to_vowel(exhale, sr, vowel=vowel)
        # f0=[(100, 1.0)]
        # f0=[(100, 1.0), (750, 0.3), (1100, 0.2), (2500, 0.01)]
        # glottal = generate_glottal_pulse(sr, ex_dur, f0)
        # glottal= breath_driven_glottal(exhale, sr, f0)
        # exhale = shape_vowel(exhale, glottal, sr, vowel="far")
        exhale = synthesize_vowel(exhale,sr)
        # exhale=
        samples.append(exhale)
        t += ex_dur

        # pause again
        pause2 = np.zeros(int(random.uniform(*pause_range) * sr))
        samples.append(pause2)
        t += len(pause2)/sr

    y = np.concatenate(samples)
    y /= np.max(np.abs(y)) + 1e-9
    return y, sr

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    y, sr = generate_breathing_loop(duration_total=20.0)
    sf.write("breathing_loop.wav", y, sr)
    print("✅ Saved: breathing_loop.wav (natural breathing loop)")
