# myTTS
* build a common ir that represent the voice generate way for all kind of language
* start with a noise gengerator to simulate the lung and Diaphragm
*  the voice sequence is a action sequence of few organ

## Dataset 
### get dataset
youtube
get audio https://turboscribe.ai/downloader/youtube/mp3/free
get text: https://notegpt.io/youtube-subtitle-downloader

### remove bgm
```
pip install demucs
python -m demucs -n htdemucs audio.wav -o output_dir
```

### split sentence
```python
from pydub import AudioSegment, silence

### Load your audio (mp3, wav, mp4 after extracting audio)
audio = AudioSegment.from_file("output_dir/vocals.wav", format="wav")

### Detect silent parts
chunks = silence.split_on_silence(
    audio,
    min_silence_len=500,     ### silence longer than 0.5s = split
    silence_thresh=-40,      ### below -40 dBFS is considered silence
    keep_silence=200         ### keep 0.2s of silence in each chunk
)

Export each chunk
for i, chunk in enumerate(chunks):
    out_file = f"output_dir/part_0/s_{i+1}.wav"
    chunk.export(out_file, format="wav")
    print("Saved:", out_file)

```
### gen text label
```python

import whisper
model = whisper.load_model("small")
for i in range(523):
    audio_path=f"output_dir/part_0/s_{i+1}.wav"
    result = model.transcribe(audio_path)
    lang=result["language"]
    text=result["text"]
    file=f"part_0/s_{i+1}.wav"
    print(f"{file}|{lang}|{text}|{0.0}")
```

### split sentence and generate text label
```python
import whisper
from pydub import AudioSegment
from zhconv import convert
model = whisper.load_model("small")
audio_path = "/data/tts/F5-TTS/dataset/zh_new/separated_2/htdemucs/zh_new_2/vocals.wav"
audio_part_idx=1
result = model.transcribe(audio_path)
audio = AudioSegment.from_file(audio_path)
text=""
s0=0
e0=0
for i, seg in enumerate(result["segments"]):
    start = seg["start"] * 1000  ### ms
    end = seg["end"] * 1000
    e0=end
    text+=seg["text"]+", "
    if e0-s0 < 3000:
        continue
        
    start=s0
    s0=end
    sentence_audio = audio[start:end]
    sentence_audio.export(f"/data/tts/F5-TTS/dataset/zh_new/part_all/sentence_{audio_part_idx}_{i+1}.wav", format="wav")
    text= text[:-1] +"。"
    simplified = convert(text, 'zh-cn')
    print(f"part_all/sentence_{audio_part_idx}_{i+1}.wav|{text}|{simplified}|{end-start}|zh")
    text=""


```

## model
### F5-tts
https://github.com/SWivid/F5-TTS
### melo-tts
https://github.com/myshell-ai

## experiment

### F5-tts
#### problem 
1. the model will speech single language with mix of multi language, there often hybrid some english when generate a chinese speech 
#### some note
1. pretrain need more resource and time to get a usable model
2. finetune is the fast way to get the usable model for different language or speaker  

