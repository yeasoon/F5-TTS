import math
import numpy as np
import time
import random

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))



class Grad:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def dot2(self, x, y):
        return self.x * x + self.y * y

    def dot3(self, x, y, z):
        return self.x * x + self.y * y + self.z * z


grad3 = [
    Grad(1, 1, 0), Grad(-1, 1, 0), Grad(1, -1, 0), Grad(-1, -1, 0),
    Grad(1, 0, 1), Grad(-1, 0, 1), Grad(1, 0, -1), Grad(-1, 0, -1),
    Grad(0, 1, 1), Grad(0, -1, 1), Grad(0, 1, -1), Grad(0, -1, -1)
]

p = [
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
    49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
]

perm = [0] * 512
gradP = [None] * 512


def seed(seed_value):
    global perm, gradP
    if 0 < seed_value < 1:
        seed_value *= 65536

    seed_value = math.floor(seed_value)

    if seed_value < 256:
        seed_value |= seed_value << 8

    for i in range(256):
        if i & 1:
            v = p[i] ^ (seed_value & 255)
        else:
            v = p[i] ^ ((seed_value >> 8) & 255)

        perm[i] = perm[i + 256] = v
        gradP[i] = gradP[i + 256] = grad3[v % 12]


# initialize seed
seed(int(time.time()))

# Skewing factors
F2 = 0.5 * (math.sqrt(3) - 1)
G2 = (3 - math.sqrt(3)) / 6


def simplex2(xin, yin):
    s = (xin + yin) * F2
    i = math.floor(xin + s)
    j = math.floor(yin + s)

    t = (i + j) * G2
    x0 = xin - i + t
    y0 = yin - j + t

    if x0 > y0:
        i1, j1 = 1, 0
    else:
        i1, j1 = 0, 1

    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1 + 2 * G2
    y2 = y0 - 1 + 2 * G2

    ii = i & 255
    jj = j & 255

    gi0 = gradP[ii + perm[jj]]
    gi1 = gradP[ii + i1 + perm[jj + j1]]
    gi2 = gradP[ii + 1 + perm[jj + 1]]

    t0 = 0.5 - x0*x0 - y0*y0
    if t0 < 0:
        n0 = 0.0
    else:
        t0 *= t0
        n0 = t0 * t0 * gi0.dot2(x0, y0)

    t1 = 0.5 - x1*x1 - y1*y1
    if t1 < 0:
        n1 = 0.0
    else:
        t1 *= t1
        n1 = t1 * t1 * gi1.dot2(x1, y1)

    t2 = 0.5 - x2*x2 - y2*y2
    if t2 < 0:
        n2 = 0.0
    else:
        t2 *= t2
        n2 = t2 * t2 * gi2.dot2(x2, y2)

    return 70.0 * (n0 + n1 + n2)


def simplex1(x):
    return simplex2(x * 1.2, -x * 0.7)


import wave
from scipy.signal import iirfilter, lfilter

class AudioSystemOffline:
    def __init__(self, block_length=512, sample_rate=44100, duration_seconds=5):
        self.block_length = block_length
        self.sample_rate = sample_rate
        self.block_time = block_length / sample_rate
        self.total_blocks = int(duration_seconds * sample_rate / block_length)

        # White noise related
        self.white_noise = self.create_white_noise(2 * sample_rate)
        self.noise_index = 0

        # Output buffer
        self.output = np.zeros(self.total_blocks * block_length, dtype=np.float32)

        # Filters
        self.aspirate_b, self.aspirate_a = self.create_biquad_bandpass(500, 0.5)
        self.fricative_b, self.fricative_a = self.create_biquad_bandpass(1000, 0.5)

    def create_white_noise(self, frame_count):
        return np.random.rand(frame_count).astype(np.float32)

    def get_noise(self, N):
        block = self.white_noise[self.noise_index:self.noise_index + N]
        self.noise_index = (self.noise_index + N) % len(self.white_noise)
        return block

    def create_biquad_bandpass(self, freq, Q):
        # compute bandwidth edges
        r = np.sqrt(1 + 1/(4*(Q**2)))
        low = freq * (r - 1/(2*Q))
        high = freq * (r + 1/(2*Q))

        # normalize to Nyquist
        low /= (self.sample_rate / 2)
        high /= (self.sample_rate / 2)

        return iirfilter(
            N=2,
            Wn=[low, high],
            btype="bandpass",
            ftype="butter"
        )

    # ---------- Main rendering loop ----------
    def render(self, Glottis, Tract):
        index = 0
        for block in range(self.total_blocks):
            noise_block = self.get_noise(self.block_length)
            aspirate = lfilter(self.aspirate_b, self.aspirate_a, noise_block)
            fricative = lfilter(self.fricative_b, self.fricative_a, noise_block)

            white_mixed = aspirate + fricative
            out = np.zeros(self.block_length, dtype=np.float32)

            for j in range(self.block_length):
                lambda1 = j / self.block_length
                lambda2 = (j + 0.5) / self.block_length

                glottal_output = Glottis.runStep(lambda1, white_mixed[j], self.sample_rate)

                vocal_output = 0
                Tract.run_step(glottal_output, white_mixed[j], lambda1, self.sample_rate)
                vocal_output += Tract.lipOutput + Tract.noseOutput

                Tract.run_step(glottal_output, white_mixed[j], lambda2, self.sample_rate)
                vocal_output += Tract.lipOutput + Tract.noseOutput

                out[j] = vocal_output * 0.125

            Glottis.finish_block(False, True)
            Tract.finish_block(self)

            self.output[index:index + self.block_length] = out
            index += self.block_length

    # ---------- Save to WAV file ----------
    def save(self, filename="output.wav"):
        audio = np.int16(self.output * 32767)
        with wave.open(filename, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(audio.tobytes())
        print(f"Saved to {filename}")

class Glottis:
    def __init__(self):
        # Time & frequency
        self.timeInWaveform = 0
        self.oldFrequency = 100
        self.newFrequency = 100
        self.UIFrequency = 100
        self.smoothFrequency = 100

        # Tenseness
        self.oldTenseness = 0.6
        self.newTenseness = 0.6
        self.UITenseness = 0.6

        # Other parameters
        self.totalTime = 0
        self.vibratoAmount = 0.005
        self.vibratoFrequency = 6
        self.intensity = 0
        self.loudness = 10
        self.isTouched = False

        # Virtual control coordinates (0-1)
        self.control_x = 0.5  # horizontal: pitch
        self.control_y = 0.5  # vertical: tenseness/loudness

        # Base note and semitone range
        self.semitones = 20
        self.baseNote = 87.3071  # F

        # Waveform
        self.frequency = self.oldFrequency
        self.waveformLength = 1.0 / self.frequency
        self.alpha = 0
        self.E0 = 1
        self.epsilon = 1
        self.shift = 0
        self.Delta = 1
        self.Te = 0
        self.omega = 1

    # -----------------------------
    # Control interface
    # -----------------------------
    def setControl(self, x: float, y: float):
        """
        x, y are normalized [0, 1]
        x: horizontal control (pitch)
        y: vertical control (tenseness)
        """
        self.control_x = clamp(x, 0, 1)
        self.control_y = clamp(y, 0, 1)

        # Calculate pitch
        semitone = self.semitones * self.control_x
        self.UIFrequency = self.baseNote * 2 ** (semitone / 12)
        if self.intensity == 0:
            self.smoothFrequency = self.UIFrequency

        # Calculate tenseness
        t = self.control_y
        self.UITenseness = 1 - math.cos(t * math.pi * 0.5)
        self.loudness = self.UITenseness ** 0.25

        # Update touch state
        self.isTouched = True

    # -----------------------------
    # Audio synthesis
    # -----------------------------
    def runStep(self, lambda_, noiseSource, sampleRate):
        timeStep = 1.0 / sampleRate
        self.timeInWaveform += timeStep
        self.totalTime += timeStep
        if self.timeInWaveform > self.waveformLength:
            self.timeInWaveform -= self.waveformLength
            self.setupWaveform(lambda_)

        out = self.normalizedLFWaveform(self.timeInWaveform / self.waveformLength)
        aspiration = self.intensity * (1 - math.sqrt(self.UITenseness)) * self.getNoiseModulator() * noiseSource
        aspiration *= 0.2 + 0.02 * simplex1(self.totalTime * 1.99)
        out += aspiration
        return out

    def getNoiseModulator(self):
        voiced = 0.1 + 0.2 * max(0, math.sin(math.pi * 2 * self.timeInWaveform / self.waveformLength))
        return self.UITenseness * self.intensity * voiced + (1 - self.UITenseness * self.intensity) * 0.3

    def normalizedLFWaveform(self, t):
        if t > self.Te:
            output = (-math.exp(-self.epsilon * (t - self.Te)) + self.shift) / self.Delta
        else:
            output = self.E0 * math.exp(self.alpha * t) * math.sin(self.omega * t)
        return output * self.intensity * self.loudness
    
    def finish_block(self, autoWobble, alwaysVoice):
        # Vibrato
        vibrato = 0.0
        vibrato += self.vibratoAmount * math.sin(2 * math.pi * self.totalTime * self.vibratoFrequency)
        vibrato += 0.02 * simplex1(self.totalTime * 4.07)
        vibrato += 0.04 * simplex1(self.totalTime * 2.15)

        if autoWobble:
            vibrato += 0.2 * simplex1(self.totalTime * 0.98)
            vibrato += 0.4 * simplex1(self.totalTime * 0.5)

        # Smooth slide toward UI frequency
        if self.UIFrequency > self.smoothFrequency:
            self.smoothFrequency = min(self.smoothFrequency * 1.1, self.UIFrequency)
        if self.UIFrequency < self.smoothFrequency:
            self.smoothFrequency = max(self.smoothFrequency / 1.1, self.UIFrequency)

        self.oldFrequency = self.newFrequency
        self.newFrequency = self.smoothFrequency * (1 + vibrato)

        # Tenseness update
        self.oldTenseness = self.newTenseness
        self.newTenseness = (
            self.UITenseness
            + 0.1 * simplex1(self.totalTime * 0.46)
            + 0.05 * simplex1(self.totalTime * 0.36)
        )

        if (not self.isTouched) and alwaysVoice:
            self.newTenseness += (3 - self.UITenseness) * (1 - self.intensity)

        # Intensity update
        if self.isTouched or alwaysVoice:
            self.intensity += 0.13
        else:
            self.intensity -= 0.05

        self.intensity = clamp(self.intensity, 0.0, 1.0)
    # -----------------------------
    # Waveform setup
    # -----------------------------
    def setupWaveform(self, lambda_):
        self.frequency = self.oldFrequency * (1 - lambda_) + self.newFrequency * lambda_
        tenseness = self.oldTenseness * (1 - lambda_) + self.newTenseness * lambda_
        self.Rd = 3 * (1 - tenseness)
        self.waveformLength = 1.0 / self.frequency

        Rd = clamp(self.Rd, 0.5, 2.7)

        Ra = -0.01 + 0.048 * Rd
        Rk = 0.224 + 0.118 * Rd
        Rg = (Rk / 4) * (0.5 + 1.2 * Rk) / (0.11 * Rd - Ra * (0.5 + 1.2 * Rk))

        Ta = Ra
        Tp = 1 / (2 * Rg)
        Te = Tp + Tp * Rk

        epsilon = 1 / Ta
        shift = math.exp(-epsilon * (1 - Te))
        Delta = 1 - shift

        RHSIntegral = (1 / epsilon) * (shift - 1) + (1 - Te) * shift
        RHSIntegral /= Delta

        totalLowerIntegral = -(Te - Tp) / 2 + RHSIntegral
        totalUpperIntegral = -totalLowerIntegral

        omega = math.pi / Tp
        s = math.sin(omega * Te)
        y = -math.pi * s * totalUpperIntegral / (Tp * 2)
        z = math.log(y)
        alpha = z / (Tp / 2 - Te)
        E0 = -1 / (s * math.exp(alpha * Te))

        self.alpha = alpha
        self.E0 = E0
        self.epsilon = epsilon
        self.shift = shift
        self.Delta = Delta
        self.Te = Te
        self.omega = omega



def move_towards(current, target, max_delta, snap_delta=None):
    """
    Move current value toward target by max_delta. If snap_delta is given,
    snap to target if within snap_delta.
    """
    delta = target - current
    if snap_delta is not None and abs(delta) < snap_delta:
        return target
    delta = clamp(delta, -max_delta, max_delta)
    return current + delta

class Tract:
    def __init__(self, n=44):
        self.n = n
        self.bladeStart = 10
        self.tipStart = 32
        self.lipStart = 39
        self.R = np.zeros(n)
        self.L = np.zeros(n)
        self.reflection = np.zeros(n + 1)
        self.newReflection = np.zeros(n + 1)
        self.junctionOutputR = np.zeros(n + 1)
        self.junctionOutputL = np.zeros(n + 1)
        self.maxAmplitude = np.zeros(n)
        self.diameter = np.zeros(n)
        self.restDiameter = np.zeros(n)
        self.targetDiameter = np.zeros(n)
        self.newDiameter = np.zeros(n)
        self.A = np.zeros(n)
        self.glottalReflection = 0.75
        self.lipReflection = -0.85
        self.lastObstruction = -1
        self.fade = 1.0
        self.movementSpeed = 15.0  # cm/s
        self.transients = []
        self.lipOutput = 0.0
        self.noseOutput = 0.0
        self.velumTarget = 0.01

    def init(self):
        self.bladeStart = int(self.bladeStart * self.n / 44)
        self.tipStart = int(self.tipStart * self.n / 44)
        self.lipStart = int(self.lipStart * self.n / 44)

        # Initialize diameters
        for i in range(self.n):
            if i < 7 * self.n / 44 - 0.5:
                diameter = 0.6
            elif i < 12 * self.n / 44:
                diameter = 1.1
            else:
                diameter = 1.5
            self.diameter[i] = self.restDiameter[i] = self.targetDiameter[i] = self.newDiameter[i] = diameter

        # Initialize reflection arrays
        self.R = np.zeros(self.n)
        self.L = np.zeros(self.n)
        self.reflection = np.zeros(self.n + 1)
        self.newReflection = np.zeros(self.n + 1)
        self.junctionOutputR = np.zeros(self.n + 1)
        self.junctionOutputL = np.zeros(self.n + 1)
        self.A = np.zeros(self.n)
        self.maxAmplitude = np.zeros(self.n)

        # Nose tract
        self.noseLength = int(28 * self.n / 44)
        self.noseStart = self.n - self.noseLength + 1
        self.noseR = np.zeros(self.noseLength)
        self.noseL = np.zeros(self.noseLength)
        self.noseJunctionOutputR = np.zeros(self.noseLength + 1)
        self.noseJunctionOutputL = np.zeros(self.noseLength + 1)
        self.noseReflection = np.zeros(self.noseLength + 1)
        self.noseDiameter = np.zeros(self.noseLength)
        self.noseA = np.zeros(self.noseLength)
        self.noseMaxAmplitude = np.zeros(self.noseLength)

        for i in range(self.noseLength):
            d = 2 * (i / self.noseLength)
            if d < 1:
                diameter = 0.4 + 1.6 * d
            else:
                diameter = 0.5 + 1.5 * (2 - d)
            diameter = min(diameter, 1.9)
            self.noseDiameter[i] = diameter

        self.newReflectionLeft = self.newReflectionRight = self.newReflectionNose = 0
        self.calculateReflections()
        self.calculateNoseReflections()
        self.noseDiameter[0] = self.velumTarget

    def run_step(self, glottalOutput, turbulenceNoise, lam, sr):
        updateAmplitudes = (random.random() < 0.1)

        # mouth
        self.processTransients(sr)
        # self.addTurbulenceNoise(turbulenceNoise)
        for index in range(self.n):
            diameter=self.diameter[index]
            self.addTurbulenceNoiseAtIndex(turbulenceNoise, index, diameter)

        # glottal reflection boundary
        self.junctionOutputR[0] = self.L[0] * self.glottalReflection + glottalOutput
        self.junctionOutputL[self.n] = self.R[self.n - 1] * self.lipReflection

        # interior junctions
        for i in range(1, self.n):
            r = self.reflection[i] * (1 - lam) + self.newReflection[i] * lam
            w = r * (self.R[i - 1] + self.L[i])
            self.junctionOutputR[i] = self.R[i - 1] - w
            self.junctionOutputL[i] = self.L[i] + w

        # nose junction
        i = self.noseStart
        r = self.newReflectionLeft * (1 - lam) + self.reflectionLeft * lam
        self.junctionOutputL[i] = r * self.R[i - 1] + (1 + r) * (self.noseL[0] + self.L[i])

        r = self.newReflectionRight * (1 - lam) + self.reflectionRight * lam
        self.junctionOutputR[i] = r * self.L[i] + (1 + r) * (self.R[i - 1] + self.noseL[0])

        r = self.newReflectionNose * (1 - lam) + self.reflectionNose * lam
        self.noseJunctionOutputR[0] = r * self.noseL[0] + (1 + r) * (self.L[i] + self.R[i - 1])

        # propagate right-moving and left-moving waves
        for i in range(self.n):
            self.R[i] = self.junctionOutputR[i] * 0.999
            self.L[i] = self.junctionOutputL[i + 1] * 0.999

            if updateAmplitudes:
                amplitude = abs(self.R[i] + self.L[i])
                if amplitude > self.maxAmplitude[i]:
                    self.maxAmplitude[i] = amplitude
                else:
                    self.maxAmplitude[i] *= 0.999

        self.lipOutput = self.R[self.n - 1]

        # nose reflections
        self.noseJunctionOutputL[self.noseLength] = self.noseR[self.noseLength - 1] * self.lipReflection

        for i in range(1, self.noseLength):
            w = self.noseReflection[i] * (self.noseR[i - 1] + self.noseL[i])
            self.noseJunctionOutputR[i] = self.noseR[i - 1] - w
            self.noseJunctionOutputL[i] = self.noseL[i] + w

        for i in range(self.noseLength):
            self.noseR[i] = self.noseJunctionOutputR[i] * self.fade
            self.noseL[i] = self.noseJunctionOutputL[i + 1] * self.fade

            if updateAmplitudes:
                amplitude = abs(self.noseR[i] + self.noseL[i])
                if amplitude > self.noseMaxAmplitude[i]:
                    self.noseMaxAmplitude[i] = amplitude
                else:
                    self.noseMaxAmplitude[i] *= 0.999

        self.noseOutput = self.noseR[self.noseLength - 1]

    def finish_block(self, AudioSystem ):
        self.reshapeTract(AudioSystem.block_time)
        self.calculateReflections()
    # -----------------------------
    # Control interface
    # -----------------------------
    def setControl(self, index, diameter, intensity=1.0):
        """
        Set the target diameter and turbulence intensity for a specific position in the tract.
        Replaces the UI touch system.
        """
        if 0 <= index < self.n:
            self.targetDiameter[index] = clamp(diameter, 0.0, 2.0)
            self.addTurbulenceNoiseAtIndex(intensity, index, diameter)

    # -----------------------------
    # Tract dynamics
    # -----------------------------
    def reshapeTract(self, deltaTime):
        amount = deltaTime * self.movementSpeed
        newLastObstruction = -1
        for i in range(self.n):
            diameter = self.diameter[i]
            target = self.targetDiameter[i]
            if diameter <= 0:
                newLastObstruction = i

            if i < self.noseStart:
                slowReturn = 0.6
            elif i >= self.tipStart:
                slowReturn = 1.0
            else:
                slowReturn = 0.6 + 0.4 * (i - self.noseStart) / (self.tipStart - self.noseStart)

            self.diameter[i] = move_towards(diameter, target, slowReturn * amount, 2 * amount)

        if self.lastObstruction > -1 and newLastObstruction == -1 and self.noseA[0] < 0.05:
            self.addTransient(self.lastObstruction)
        self.lastObstruction = newLastObstruction

        # Nose
        self.noseDiameter[0] = move_towards(self.noseDiameter[0], self.velumTarget, amount * 0.25, amount * 0.1)
        self.noseA[0] = self.noseDiameter[0] ** 2

    def calculateReflections(self):
        self.A = self.diameter ** 2
        for i in range(1, self.n):
            self.reflection[i] = self.newReflection[i]
            if self.A[i] == 0:
                self.newReflection[i] = 0.999
            else:
                self.newReflection[i] = (self.A[i-1] - self.A[i]) / (self.A[i-1] + self.A[i])

        # Junction with nose
        self.reflectionLeft = self.newReflectionLeft
        self.reflectionRight = self.newReflectionRight
        self.reflectionNose = self.newReflectionNose
        sumA = self.A[self.noseStart] + self.A[self.noseStart + 1] + self.noseA[0]
        self.newReflectionLeft = (2 * self.A[self.noseStart] - sumA) / sumA
        self.newReflectionRight = (2 * self.A[self.noseStart + 1] - sumA) / sumA
        self.newReflectionNose = (2 * self.noseA[0] - sumA) / sumA

    def calculateNoseReflections(self):
        self.noseA = self.noseDiameter ** 2
        for i in range(1, self.noseLength):
            self.noseReflection[i] = (self.noseA[i-1] - self.noseA[i]) / (self.noseA[i-1] + self.noseA[i])

    # -----------------------------
    # Transients
    # -----------------------------
    def addTransient(self, position):
        trans = {
            "position": position,
            "timeAlive": 0.0,
            "lifeTime": 0.2,
            "strength": 0.3,
            "exponent": 200
        }
        self.transients.append(trans)

    def processTransients(self, sampleRate):
        for trans in self.transients:
            amplitude = trans["strength"] * 2 ** (-trans["exponent"] * trans["timeAlive"])
            self.R[trans["position"]] += amplitude / 2
            self.L[trans["position"]] += amplitude / 2
            trans["timeAlive"] += 1.0 / (sampleRate * 2)
        self.transients = [t for t in self.transients if t["timeAlive"] <= t["lifeTime"]]

    # -----------------------------
    # Turbulence noise
    # -----------------------------

    def addTurbulenceNoiseAtIndex(self, turbulenceNoise, index, diameter):
        i = int(index)
        delta = index - i
        # Use Glottis.getNoiseModulator() externally
        thinness0 = clamp(8 * (0.7 - diameter), 0, 1)
        openness = clamp(30 * (diameter - 0.3), 0, 1)
        noise0 = turbulenceNoise * (1 - delta) * thinness0 * openness
        noise1 = turbulenceNoise * delta * thinness0 * openness
        if i + 1 < self.n:
            self.R[i + 1] += noise0 / 2
            self.L[i + 1] += noise0 / 2
        if i + 2 < self.n:
            self.R[i + 2] += noise1 / 2
            self.L[i + 2] += noise1 / 2

audio_offline=AudioSystemOffline()
tract=Tract()
tract.init()
glottal=Glottis()

audio_offline.render(glottal, tract)
audio_offline.save()