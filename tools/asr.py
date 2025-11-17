import whisper

# Load model (options: tiny, base, small, medium, large)
# larger models = better accuracy but slower
model = whisper.load_model("small")

# Path to your audio file (wav, mp3, m4a, etc.)
# audio_path = "/data/tts/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_zh.wav"
# audio_path="/data/tts/F5-TTS/dataset/ru/wavs/early_short_stories/early_short_stories_0033.wav"
audio_path="/data/tts/F5-TTS/tests/infer_cli_basic_new.wav"
# audio_path="/data/tts/output.wav"

# Transcribe the speech to text
result = model.transcribe(audio_path)

print("Detected language:", result["language"])
print("Transcription:", result["text"])
# Moi non plus. Il alluma une cigarette à l'aide d'une allumette-bougie qui l'agita plusieurs fois pour l'éteindre.
# Moi non plus. Il alluma une cigarette à l’aide d’une allumette-bougie qui l'agita plusieurs fois pour l’éteindre.
# Девицу Тамару, жилище, который лежит за пределами И ерусалима. И я хочу знать, правда ли говорит о виновам?
# девицу Тамару, жилище, который лежит за пределами и Ярусалима, и я хочу знать правду ли говорит о веном.
# Девицу Тамару, жилище, который лежит за пределами Ерус Валима. Е е хочу инать Правду ли говорит о винаем?
# девицу Тамару, жилище  которой лежит за пределами И ерусалима. И я хочу знать правду ли говорит А виноам?
# 9 Сутамару, Елещепа Той лежит за предлами Ярусалима и окачу ленат, правда ли говорит Хайинам.