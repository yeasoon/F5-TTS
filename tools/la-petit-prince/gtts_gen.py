
def split_line(input_path):
    import re
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile(r'(\n)', re.IGNORECASE)
    parts = pattern.split(content)
    all_lines=[]
    lines=""
    for i in range(0, len(parts), 2):
        body = parts[i].strip() if i < len(parts) else ""
        if body:
            body=body.replace("«", "")
            body=body.replace("»", "")
            all_lines.append(body)
            lines+=body
    return all_lines, lines

from pydub import AudioSegment
def generate(text, language="fr",idx=0):
    from gtts import gTTS

    # # Your Spanish text
    # text = """
    # La candente mañana de febrero en que Beatriz Viterbo murió, después de una imperiosa agonía que no se rebajó un solo instante ni al sentimentalismo ni al miedo, 
    # """

    # Generate audio
    text=f"""{text}"""
    print(text)
    
    tts = gTTS(text=text, lang=language)
    tmp_file=f"/data/tts/la-petit-prince/audio/{idx}_tmp.mp3"
    tts.save(tmp_file)
    return AudioSegment.from_mp3(tmp_file)

    return 
def tts_model():
    from TTS.api import TTS
    import random
    import numpy as np
    file_path="output.wav"
    all_lines, lines=split_line("/data/tts/la-petit-prince/ch1.txt")
    all_wav=None
    silence_duration_ms =  random.uniform(0.8, 1.2)*1000
    for idx, item in enumerate(all_lines):
        wav_=generate(item, language="fr",idx=idx)
        silence = AudioSegment.silent(duration=silence_duration_ms)
        if idx ==0:
            all_wav=wav_+silence
            continue
        all_wav = all_wav + silence + wav_


            
    all_wav.export(file_path, format="wav")
    print(file_path)

   
tts_model()