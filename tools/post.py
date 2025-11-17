import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfilt
# -----------------------------
# Parameters
# -----------------------------
input_file = "/data/tts/tests/infer_cli_basic.wav"         # Input audio file
output_file = "/data/tts/tests/infer_cli_basic_cl.wav"  # Output cleaned audio
def amplify_mid_freq(input_file, output_file):
    sr = None  # None = keep original sample rate
    low_freq = 30    # Low cut for mid frequencies (Hz)
    high_freq = 2500  # High cut for mid frequencies (Hz)
    gain = 1.5        # Amplification factor for mid frequencies

    # -----------------------------
    # Load audio
    # -----------------------------
    y, sr = librosa.load(input_file, sr=sr)

    # -----------------------------
    # Design a band-pass filter for mid frequencies
    # -----------------------------
    sos = butter(N=4, Wn=[low_freq, high_freq], btype='band', fs=sr, output='sos')

    # Apply filter
    mid = sosfilt(sos, y)

    # -----------------------------
    # Combine original + boosted mid frequencies
    # -----------------------------
    y_boosted = y + gain * mid
    # Normalize to avoid clipping
    y_boosted = y_boosted / np.max(np.abs(y_boosted))

    # -----------------------------
    # Optional: trim silences
    # -----------------------------
    # y_final, _ = librosa.effects.trim(y_boosted, top_db=20)
    y_final = y_boosted


    n_fft = 1024
    hop_length = 256
    D = librosa.stft(y_final, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(D), np.angle(D)

    # Frequency smoothing (along freq axis)
    mag_smooth = gaussian_filter(mag, sigma=(1,0))  # smooth high freq spikes

    # Optional: mild temporal smoothing (avoid robotic lag)
    mag_smooth = gaussian_filter(mag_smooth, sigma=(0,0.5))  

    # Reconstruct waveform
    y_recon = librosa.istft(mag_smooth * np.exp(1j * phase), hop_length=hop_length)

    # Normalize
    y_recon = y_recon / np.max(np.abs(y_recon))
    y_final = y_recon
    def lowpass_filter(audio, sr, cutoff=8000):
        # 4th-order Butterworth filter
        sos = butter(4, cutoff, btype='low', fs=sr, output='sos')
        return sosfilt(sos, audio)

    # Apply
    y_filtered = lowpass_filter(y_final, sr, cutoff=8000)  # 8 kHz cutoff for speech

    # y_final = y_filtered
    # -----------------------------
    # Save result
    # -----------------------------
    sf.write(output_file, y_final, sr)
    print(f"Mid-frequency boosted audio saved to {output_file}")
def silence_part(input_file, output_file):
    sr = None
    min_silence_sec = 0.3   # consider silence longer than 0.3s
    fade_duration = 0.2     # fade duration in seconds
    top_db = 30              # threshold for silence detection

    # -----------------------------
    # Load audio
    # -----------------------------
    y, sr = librosa.load(input_file, sr=sr)
    fade_samples = int(fade_duration * sr)
    min_silence_samples = int(min_silence_sec * sr)

    # -----------------------------
    # Detect non-silent regions
    # -----------------------------
    non_silent = librosa.effects.split(y, top_db=top_db)

    # -----------------------------
    # Helper: find nearest zero-crossing
    # -----------------------------
    def nearest_zero_cross(samples):
        zero_crossings = np.where(np.diff(np.sign(samples)))[0]
        if len(zero_crossings) == 0:
            return 0
        return zero_crossings[-1]

    # -----------------------------
    # Process segments with smooth transitions
    # -----------------------------
    y_new = []

    for i in range(len(non_silent)):
        start, end = non_silent[i]
        segment = y[start:end]
        
        # Append current segment
        y_new.append(segment)
        
        # If not last segment, create smooth transition
        if i < len(non_silent) - 1:
            next_start, _ = non_silent[i+1]
            gap = next_start - end
            
            if gap > min_silence_samples:
                # Extract last part of current and first part of next
                fade_out_start = max(end - fade_samples, start)
                fade_out = y[fade_out_start:end]
                fade_out = fade_out[nearest_zero_cross(fade_out):]  # align to zero-cross
                
                fade_in = y[non_silent[i+1][0]:non_silent[i+1][0]+fade_samples]
                fade_in = fade_in[:nearest_zero_cross(fade_in)+1]     # align to zero-cross
                
                # Hann window crossfade
                N = min(len(fade_out), len(fade_in))
                window = np.hanning(2*N)
                fade_out_win = window[:N]
                fade_in_win = window[N:]
                transition = fade_out[:N]*fade_out_win + fade_in[:N]*fade_in_win
                
                # Append transition
                y_new.append(transition)

    # -----------------------------
    # Concatenate and normalize
    # -----------------------------
    y_final = np.concatenate(y_new)
    y_final = y_final / np.max(np.abs(y_final))

    # -----------------------------
    # Save result
    # -----------------------------
    sf.write(output_file, y_final, sr)
    print(f"Smooth audio with natural transitions saved to {output_file}")
silence_part("/data/tts/tests/infer_cli_basic.wav","/data/tts/tests/infer_cli_basic_cl.wav")
amplify_mid_freq("/data/tts/tests/infer_cli_basic_cl.wav","/data/tts/tests/infer_cli_basic_cl1.wav")
