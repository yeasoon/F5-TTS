# import os
# import librosa
# import numpy as np
# import soundfile as sf
# from tqdm import tqdm
# import parselmouth  # for Praat-based pitch extraction

# # === Paths ===
# AUDIO_DIR = "/data/tts/F5-TTS/dataset/zh_new/part_0"
# TEXT_DIR = "/data/tts/F5-TTS/dataset/zh_new/part_0/transcripts"
# FEATURE_DIR = "/data/tts/F5-TTS/dataset/zh_new/part_0/features"

# os.makedirs(FEATURE_DIR, exist_ok=True)

# # === Function to extract F0 using Parselmouth ===
# def extract_f0(audio_path):
#     snd = parselmouth.Sound(audio_path)
#     pitch = snd.to_pitch()
#     f0_values = pitch.selected_array['frequency']
#     f0_values[f0_values==0] = np.nan  # mark unvoiced frames
#     return f0_values

# # === Function to extract RMS ===
# def extract_rms(y):
#     rms = librosa.feature.rms(y=y)
#     return rms.flatten()

# # === Function to load audio ===
# def load_audio(audio_path, sr=22050):
#     y, _ = librosa.load(audio_path, sr=sr)
#     return y

# # === Main extraction loop ===
# for file_name in tqdm(os.listdir(AUDIO_DIR)):
#     if not file_name.endswith(".wav"):
#         continue
#     base_name = os.path.splitext(file_name)[0]
#     audio_path = os.path.join(AUDIO_DIR, file_name)
#     text_path = os.path.join(TEXT_DIR, base_name + ".txt")
    
#     # if not os.path.exists(text_path):
#     #     print(f"Transcript missing for {file_name}, skipping.")
#     #     continue
    
#     # Load audio
#     y = load_audio(audio_path)
    
#     # Extract features
#     f0 = extract_f0(audio_path)
#     rms = extract_rms(y)
#     print(f0)
#     print(rms)
#     break
#     # Save features
#     np.save(os.path.join(FEATURE_DIR, base_name + "_f0.npy"), f0)
#     np.save(os.path.join(FEATURE_DIR, base_name + "_rms.npy"), rms)
#     break
    
#     # # Copy transcript
#     # with open(text_path, "r", encoding="utf-8") as f:
#     #     text = f.read().strip()
#     # with open(os.path.join(FEATURE_DIR, base_name + "_txt.txt"), "w", encoding="utf-8") as f:
#     #     f.write(text)


# import librosa
# import numpy as np
# import scipy.signal

# y, sr = librosa.load("/data/tts/F5-TTS/dataset/zh_new/part_all/sentence_0_11.wav", sr=22050)
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)          # 13 维
# print(mfcc.shape)
# f0 = librosa.yin(y=y, fmin=75, fmax=600, sr=sr)         # 帧级 F0
# # print(f0)
# zcr = librosa.feature.zero_crossing_rate(y=y)
# import librosa
# import numpy as np
# from scipy.signal import iirfilter, lfilter
# import soundfile as sf

# def apply_eq(y, sr, freq, gain_db, q=1.0):
#     A = 10**(gain_db / 40)
#     w0 = 2 * np.pi * freq / sr
#     alpha = np.sin(w0) / (2 * q)

#     # Peaking EQ filter
#     b = [1 + alpha*A, -2*np.cos(w0), 1 - alpha*A]
#     a = [1 + alpha/A, -2*np.cos(w0), 1 - alpha/A]

#     return lfilter(b, a, y)

# # y, sr = librosa.load("audio.wav", sr=22050)
# # y_eq = apply_eq(y, sr, freq=3500, gain_db=6)  # boost speech presence

# y_eq = apply_eq(y, sr, freq=1200, gain_db=6)
# sf.write("output_eq.wav", y_eq, sr)
# print(zcr)
# # Compute short-time energy
# frame_length = 512
# hop_length = 128
# energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

# # Smooth energy
# energy_smooth = scipy.signal.medfilt(energy, kernel_size=5)

# # Threshold to detect voiced segments
# threshold = np.mean(energy_smooth) * 1.2
# voiced_frames = energy_smooth > threshold

# # Convert frames to time
# times = librosa.frames_to_time(np.arange(len(energy_smooth)), sr=sr, hop_length=hop_length)

# # Find segments
# segments = []
# start = None
# for t, voiced in zip(times, voiced_frames):
#     if voiced and start is None:
#         start = t
#     elif not voiced and start is not None:
#         end = t
#         segments.append((start, end))
#         start = None
# if start is not None:
#     segments.append((start, times[-1]))

# print(f"Detected segments (approx. words/phonemes): {len(segments)}")
# for s in segments:
#     print(s, "duration:", s[1]-s[0])


# bamboo_pipe_wav.py
# flute_wav_pde.py
import numpy as np
import wave

# ========== 物理/数值参数 ==========
L   = 2.52          # 管长（m）
c   = 343.0         # 声速（m/s）
N   = 500           # 空间分段
dx  = L / N
dt  = 0.1 * dx / c  # CFL 稳定条件
FS  = int(1/dt)     # 等效采样率
DUR = 3.0           # 模拟时长（s）
NT  = int(DUR*FS)

OUT_POS = N//4      # 取输出位置（离吹孔 1/4 处）
WAV_PATH = "bamboo_flute.wav"

# ========== 初始化 ==========
p   = np.zeros(N+1)      # 当前压力
p1  = np.zeros_like(p)   # t-dt
src = np.zeros_like(p)

# 窄带激励：40-sample 的 Ricker 小波，中心 800 Hz
t0  = 40*dt
f0  = 800
for n in range(40):
    t = n*dt
    src[n] = (1 - 2*(np.pi*f0*(t-t0))**2) * np.exp(-(np.pi*f0*(t-t0))**2)
src *= 0.1

# ========== FDTD 主循环 ==========
out = []
for n in range(NT):
    # 1. 空间二阶差分
    d2p = (p[2:] - 2*p[1:-1] + p[:-2]) / dx**2
    # 2. 更新内部节点
    p_new = 2*p[1:-1] - p1[1:-1] + (c*dt)**2 * d2p
    # 3. 注入源（吹孔在 x=0）
    p_new[0] += src[n] if n<len(src) else 0.0
    # 4. 边界条件：两端开口 → p=0
    p_new = np.concatenate([[0], p_new, [0]])
    # 5. 轮转
    p1, p = p, p_new
    # 6. 记录输出
    out.append(p[OUT_POS])

# ========== 保存 WAV ==========
out = np.array(out)
out /= np.max(np.abs(out))          # 归一化
out16 = (out * 32767).astype(np.int16)
with wave.open(WAV_PATH, 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(FS)
    w.writeframes(out16.tobytes())

print(f"已保存 {WAV_PATH}  等效采样率={FS} Hz")