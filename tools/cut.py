from pydub import AudioSegment

# Load MP3
audio = AudioSegment.from_mp3("./barackobama2004dncARXE.mp3")

# Define start and end times (milliseconds)
# Define time in (hour, minute, second)
start_time = (0, 1, 39)  # 0h 1m 23s
end_time   = (0, 1, 45)   # 0h 2m 5s

# Convert to milliseconds
def hms_to_ms(h, m, s):
    return (h * 3600 + m * 60 + s) * 1000

start_ms = hms_to_ms(*start_time)
end_ms = hms_to_ms(*end_time)

# Cut and export
segment = audio[start_ms:end_ms]
segment.export("speech_0.wav", format="wav")
print("✅ Saved: speech_0.wav")
