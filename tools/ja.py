from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf

# Load the pre-trained model
text2speech = Text2Speech.from_pretrained("kan-bayashi/jsut_vits_prosody")

# Input Japanese text
text = "おはようございます。今日はとてもいい天気ですね。"

# Generate speech
speech = text2speech(text)["wav"]

# Save to WAV file
sf.write("output.wav", speech.numpy(), text2speech.fs, "PCM_16")
