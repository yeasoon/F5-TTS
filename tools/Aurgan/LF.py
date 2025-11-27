import numpy as np


def lf_waveform(F0=140, Rd=1.2, sample_rate=48000, duration=3.0):
    """
    Generate an audible Liljencrants–Fant (LF) glottal flow derivative
    for a given duration in seconds.
    """

    # ========== Convert Rd to LF parameters ==========
    T0 = 1.0 / F0
    samples_per_cycle = int(T0 * sample_rate)

    # Convert Rd → LF parameters
    Rd = max(0.3, min(Rd, 2.7))  # safe clamp
    Ra = -1.0 + 4.8 * Rd
    Rk = 22.0 + 0.04 / Rd
    Rg = (2.0 - Ra) / (1.0 + Rk)

    Tp = T0 / (1.0 + Rk)
    Te = Tp + Rg * (T0 - Tp)
    Ta = Ra * T0

    # LF constants
    w = np.pi / Tp
    eps = 1.0 / Ta
    E0 = -np.sin(w * Tp) * np.exp(eps * (Tp - Te))

    # ---- Generate one cycle ----
    t = np.linspace(0, T0, samples_per_cycle, endpoint=False)
    one_cycle = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < Tp:
            one_cycle[i] = np.sin(w * ti)
        elif ti < Te:
            one_cycle[i] = E0 * np.exp(-eps * (ti - Tp))
        else:
            one_cycle[i] = 0.0

    # Remove DC from one cycle
    one_cycle -= np.mean(one_cycle)

    # ---- Repeat cycles to fill duration ----
    total_samples = int(sample_rate * duration)
    num_cycles = int(np.ceil(total_samples / samples_per_cycle))

    wave = np.tile(one_cycle, num_cycles)
    wave = wave[:total_samples]  # crop to exact duration

    # Normalize
    wave /= np.max(np.abs(wave)) + 1e-8
    return wave.astype(np.float32)

sr=48000
wave = lf_waveform(F0=140, Rd=1.0, sample_rate=sr)

import soundfile as sf
sf.write("output_LF.wav", wave, sr)