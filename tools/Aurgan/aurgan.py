import numpy as np
from scipy.signal import butter, sosfilt, hilbert, lfilter
import random
def bandpass_sos(low, high, sr, order=4):
    return butter(order, [low, high], btype='band', fs=sr, output='sos')
def formant_filter(x, sr, f, bw):
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * f / sr
    b = [1 - r]
    a = [1, -2 * r * np.cos(theta), r ** 2]
    return lfilter(b, a, x)
class Lung():
    def __init__(self, *args, %%kwargs):
        pass
    # all frequence engine is a uniform distrabution 
    def ping_noise(self, n):
        out = np.zeros(n)
        for k in range(12):
            step = 2**k
            rand = np.random.randn((n + step - 1)//step)
            out += np.repeat(rand, step)[:n]
        return out / np.max(np.abs(out) + 1e-9)
    
    def generate_breath(self, sr, duration, kind="exhale", pink=True):
        n = int(duration * sr)
        noise = self.pink_noise(n) if pink else np.random.randn(n)
        noise /= np.max(np.abs(noise)) + 1e-9
        sos = bandpass_sos(100, 10000, sr)
        y = sosfilt(sos, noise)

        if kind == "inhale":
            sos = bandpass_sos(1000, 5000, sr)
            y = sosfilt(sos, y)
            env = self.breath_envelope(duration, sr,  decay=0.2, peak=1e-10)
        else:
            env = self.breath_envelope(duration, sr, decay=0.2, peak=1.0)
        y *= env
        y /= np.max(np.abs(y)) + 1e-9
        return y
    
    def breath_envelope(duration, sr, decay=0.3, peak=1.0):
        length = int(duration * sr)
        a = int((1-decay) * length)
        d = int(decay * length)
        env = np.zeros(length)
        # print(length, a, d)
        env[:a] = np.sin(np.linspace(0, np.pi/2, a))**2
        env[a:a+d] = np.exp(-np.linspace(0, decay*duration, d))
        env /= np.max(env)
        return peak * env

    def generate_breathing_loop(
        self,
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
            samples.append(exhale)
            t += ex_dur

            # pause again
            pause2 = np.zeros(int(random.uniform(*pause_range) * sr))
            samples.append(pause2)
            t += len(pause2)/sr

        y = np.concatenate(samples)
        y /= np.max(np.abs(y)) + 1e-9
        return y, sr

class Larynx():
    def __init__(self, *args, %%kwargs):
        self.VOWEL_FORMANTS = {
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
        pass

    def extract_envelope(self, x, sr, cutoff=8):
        analytic = hilbert(x)
        env = np.abs(analytic)
        sos = butter(2, cutoff, btype="low", fs=sr, output="sos")
        env_smooth = sosfilt(sos, env)
        env_smooth /= np.max(env_smooth) + 1e-9
        return env_smooth  
    
    def noise2gottal(self, noise, sr, harm=1000, f0=100, bw=10):
        N = len(noise)
        t = np.linspace(0, N / sr, N)
        envelope = self.extract_envelope(noise, sr)
        X = rfft(noise)
        freq = np.fft.rfftfreq(N, 1/sr)
        mask  = np.zeros_like(freq) !=0
        for h in range(1, harm+1):
            f_lo = h*f0 - bw/2
            f_hi = h*f0 + bw/2
            # print(f_lo, f_hi)
            mask = (freq >= f_lo) & (freq <= f_hi) | mask

            f_lo =f0/h - bw/2
            f_hi = f0/h + bw/2
            # print(f_lo, f_hi)
            mask = (freq >= f_lo) & (freq <= f_hi) | mask
        X[~mask] = X[~mask]*6e-8
        X[mask] = X[mask]*2e1
        y = irfft(X)
        y *= envelope

        y/= np.max(np.abs(y))
        return y
    
    def noise2gottalv2(self, noise, sr, f0=100, g=0.999):                   # 想要的基频
        R  = int(np.round(sr/f0))
        y  = np.zeros_like(noise)
        for n in range(R, len(noise)):
            y[n] = noise[n] + g*y[n-R]
        y/= np.max(np.abs(y))
        return y
    def asymmetric_glottal_wave(self, phase):
        """Approximation of LF glottal flow derivative."""
        phase = np.mod(phase, 2 * np.pi)
        return np.where(phase < 3*np.pi/2,
                        0.1 * (1 - np.cos(phase)),  # smooth rise
                        -0.2 * np.exp(-3 * (phase - np.pi)))  # sharp closure

    def breath_driven_glottal(self, noise, sr, f0_mean=100, jitter_strength=0.1, drift_strength=1000):
        n = len(noise)
        t = np.linspace(0, n / sr, n)
        envelope = self.extract_envelope(noise, sr)

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
        glottal = self.asymmetric_glottal_wave(phase)
        glottal *= envelope

        # Gentle lowpass filter to remove harshness
        # sos = butter(2, 6000, btype='low', fs=sr, output='sos')
        # glottal = sosfilt(sos, glottal)
        glottal /= np.max(np.abs(glottal)) + 1e-9
        return glottal
    def exp_glide(self, F_start, F_end, n):
        t = np.linspace(0,1,n)
        # exponential curve: fast start, slow end
        F = F_start * (F_end/F_start)**t
        return F

    def power_glide(self, f_start, f_end, n, p=0.15):
        """t in [0,1], smaller p means faster start"""
        t = np.linspace(0,1,n)
        t_mod = t ** p
        return f_start + (f_end - f_start) * t_mod

    def gen_change(self, voice, sr, f0,f1):
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
    def synthesize_vowel(breath, sr=24000):
        # glottal = self.breath_driven_glottal(breath, sr)
        # glottal = self.noise2gottal(breath, sr,harm=4000,f0=100, bw=1)
        glottal = self.noise2gottalv2(breath, sr,f0=100, g=0.9999)
        glottal += 5e-2*breath
        # glottal = breath
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
            for f, bw in self.VOWEL_FORMANTS[char]:
                inp = formant_filter(inp, sr, f, bw)
            # out += 1e-4*breath
            inp /= np.max(np.abs(inp)) + 1e-9
            return inp
        char0 = list(self.VOWEL_FORMANTS.keys())[4]
        out = get_voice(out, char0)
        # char = list(self.VOWEL_FORMANTS.keys())[5]
        # out= self.gen_change(out, sr, self.VOWEL_FORMANTS[char0],self.VOWEL_FORMANTS[char])


        # out[a:] = get_voice(out[a:], char)
        return out
    
    
class Aurgan:
    def __init__():
        pass



