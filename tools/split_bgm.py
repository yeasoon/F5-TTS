from spleeter.separator import Separator

# 2stems model = [vocals, accompaniment]
separator = Separator('spleeter:2stems')

separator.separate_to_file('/data/tts/F5-TTS/zh_new.wav', '/data/tts/F5-TTS/dataset/zh_new')
