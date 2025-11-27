# vad.py
import collections
import webrtcvad
import numpy as np

# frame helper
class Frame:
    def __init__(self, bytes_, timestamp, duration):
        self.bytes = bytes_
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    Yield Frames of PCM16 bytes from float32 numpy audio [-1,1]
    """
    assert sample_rate in (8000,16000,32000,48000), "webrtcvad supports 8/16/32/48k"
    n_bytes_per_sample = 2
    audio_int16 = np.int16(np.clip(audio, -1, 1) * 32767)
    byte_data = audio_int16.tobytes()
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * n_bytes_per_sample)
    offset = 0
    ts = 0.0
    duration = frame_duration_ms / 1000.0
    while offset + frame_size <= len(byte_data):
        yield Frame(byte_data[offset:offset + frame_size], ts, duration)
        ts += duration
        offset += frame_size

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad_mode, frames):
    """
    Generator that yields (start, end) in seconds for speech segments.
    Uses sliding window; returns merged segments.
    """
    vad = webrtcvad.Vad(int(vad_mode))  # 0-3 (aggressiveness)
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_windows = []
    start_time = None

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start_time = ring_buffer[0][0].timestamp
                # drain ring buffer
                ring_buffer.clear()
        else:
            # in speech
            voiced_windows.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                end_time = frame.timestamp + frame.duration
                yield (start_time, end_time)
                triggered = False
                ring_buffer.clear()
                voiced_windows = []
    # end
    if triggered and start_time is not None:
        # finish last
        last_frame = frame
        yield (start_time, last_frame.timestamp + last_frame.duration)
# energy_seg.py
import numpy as np
import librosa
def silence_segments(path, min_silence_len=400, silence_thresh=-40, keep_silence=200):
    from pydub import AudioSegment, silence
    from pydub.silence import detect_nonsilent

    ### Load your audio (mp3, wav, mp4 after extracting audio)
    audio = AudioSegment.from_file(path, format="wav")

    ### Detect silent parts
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,     ### silence longer than 0.5s = split
        silence_thresh=silence_thresh,      ### below -40 dBFS is considered silence
        keep_silence=keep_silence         ### keep 0.2s of silence in each chunk
    )
    # # Track timestamps
    # timestamps = []
    # current_pos = 0  # in ms

    # for chunk in chunks:
    #     start = audio[current_pos:].find(chunk) + current_pos
    #     end = start + len(chunk)
    #     timestamps.append({"start": start, "end": end})
    #     current_pos = end
    # # for i, chunk in enumerate(chunks):
    # #     # print(dir(chunk))
    # #     # print("Duration (ms):", len(chunk))
    # #     print("Duration (sec):", len(chunk) / 1000)
    #     # out_file = f"/data/tts/F5-TTS/tools/analysis/tmp/s_{i+1}.wav"
    #     # chunk.export(out_file, format="wav")
    #     # break

    # Detect non-silent regions (returns start/end in ms)
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,   # minimum silence length in ms
        silence_thresh=silence_thresh,     # threshold in dBFS
        seek_step=1
    )
    timestamps = []

    # Print timestamps
    for i, (start, end) in enumerate(nonsilent_ranges):
        # print(f"Chunk {i}: start={start/1000:.2f}s, end={end/1000:.2f}s")
        timestamps.append({"start": start, "end": end})
    return chunks, timestamps, audio
def energy_segments(audio, sr, frame_length=1024, hop_length=256, top_db=30, min_speech_ms=150):
    """
    energy-based non-speech/speech segmentation using librosa.effects.split
    returns list of (start_s, end_s)
    """
    # input audio is float32
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    results = []
    for i, (s, e) in enumerate(intervals):
        start_s = float(s) / sr
        end_s = float(e) / sr
        if (end_s - start_s) * 1000 >= min_speech_ms:
            results.append((start_s, end_s))
    return results

def merge_segments(seg_a, seg_b, max_gap_s=0.25):
    """
    Merge two lists of segments, combine overlapping and close ones.
    seg_a, seg_b: lists of (start,end)
    """
    segs = sorted(list(seg_a) + list(seg_b), key=lambda x: x[0])
    if not segs:
        return []
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        last = merged[-1]
        if s <= last[1] + max_gap_s:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return [(float(s), float(e)) for s, e in merged]

# features.py
import librosa
import numpy as np

def extract_mel(audio, sr, n_mels=80, hop_length=256, n_fft=1024):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=1.0)
    # convert to log (dB)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel  # shape (n_mels, T)

def extract_energy(audio, frame_length=1024, hop_length=256):
    # RMS energy per frame
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return energy

def extract_pitch(audio, sr, fmin=50.0, fmax=500.0, hop_length=256):
    # using librosa.yin
    pitches = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length, frame_length=1024)
    # yin gives float Hz or np.nan when unvoiced; keep that
    return pitches

# utils.py
import soundfile as sf
import numpy as np

def load_audio(path, sr=16000):
    audio, file_sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mixdown
    if file_sr != sr:
        import librosa
        # audio = librosa.resample(audio, file_sr, sr)
        audio = librosa.resample(y=audio, orig_sr=file_sr, target_sr=sr)
    return audio, sr

def save_segments_json(segments, out_path):
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def segment_audio_file(path, out_json=None,
                       sr=16000,
                       vad_mode=2,
                       frame_duration_ms=30,
                       padding_duration_ms=300,
                       energy_top_db=30,
                       merge_max_gap=0.2):
    chunks, timestamps, audio=silence_segments(path,min_silence_len=10, keep_silence=5)
    sr=audio.frame_rate
    # audio, sr = load_audio(path, sr=sr)
    # # 1) VAD-based coarse segments
    # frames = list(frame_generator(frame_duration_ms, audio, sr))
    # vad_segs = list(vad_collector(sr, frame_duration_ms, padding_duration_ms, vad_mode, frames))

    # # 2) Energy-based segments
    # energy_segs = energy_segments(audio, sr, top_db=energy_top_db)

    # # 3) Merge
    # merged = merge_segments(vad_segs, energy_segs, max_gap_s=merge_max_gap)

    # 4) Extract features per segment
    results = {
        "audio_file": path,
        "duration": float(len(audio)),
        "segments": []
    }
    sum=0
    for i, chunk in enumerate(chunks):
        sum+=len(chunk)
        s_idx = int(timestamps[i]["start"])
        e_idx = int(timestamps[i]["end"])
        seg_audio = np.array(chunk.get_array_of_samples()).astype(np.float32)/2 ** (8 * chunk.sample_width - 1)
        # features
        # mel = extract_mel(seg_audio, sr,n_mels=50, hop_length=64, n_fft=128)
        energy = extract_energy(seg_audio)
        pitch = extract_pitch(seg_audio, sr)
        # summary stats
        seg_obj = {
            "id": i,
            "start": float(s_idx),
            "end": float(e_idx),
            "duration": [float(e_idx - s_idx), len(chunk)],
            "features": {
                # "mel_frames": mel.shape[1],
                "energy_mean": float(np.mean(energy)) if energy.size else 0.0,
                "pitch_median": float(np.nanmedian(pitch)) if pitch.size else None
            }
        }
        results["segments"].append(seg_obj)
    print(sum)
    print(results)
    if out_json:
        save_segments_json(results, out_json)
    return results
# segment_audio_file("/data/tts/F5-TTS/dataset/zh_new/part_all/sentence_0_1.wav")
segment_audio_file("/data/tts/F5-TTS/dataset/zh/wavs/call_to_arms/call_to_arms_0009.wav")