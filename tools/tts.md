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

#Export each chunk
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
### Diversity of the data
* speaker
    * physical
        * gender
        * speed custom
        * engergy custom
        * pitch custom
        * duration custom

    * psychological
        * emotion
        * language 
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

#### melspectrogram
model predict melspectrogram, by change melspectrogram can change the audio generated
##### change melspectrogram
1. enlarge the sound: mel+x, mel is a exp coff
2. speed up the sound: sample/interpolate mel in time dim
3. EQ can only small change the sould
4. some high level control is complicated on melspectrogram
#### ref audio and text
1. concat and projection is more valid than add to embeding
2. ref audio can control the emotion, gender, speed and some other high level part
3. the human like key part is how to generate the high level sequence of those feature
#### some key sequence
1. paragrah+duration
    * human read seperation, no need for audio or speech
2. sentence+duration
    * base unit of a speech seperation, emotion gender, speed and other hight level feature apply on this level
3. word+duration
    * seperation for a collection of big change of phonemes
4. character+duration
    * seperation of read text
5. phonemes+duration
    * unit of constant mel part
6. mel_cut+duration
    * speech base feature

## pink trombone
### vocal tract 
* pharyngeal cavity
* oral cavity
* nasal cavity 

### signal filter
#### time domain
```python
y[n]=-1*sum(a[k]*y[n-k])+sum(b[k]*x[n-k])
y=-1*conv0(y)+conv1(x)
```

#### frequency domain
```python
X=Me
H=(Be)/(1+Ae)
Y=H*X
```

#### wave function
##### D’Alembert exact solution
```
p(x,t) = pr(x−ct) + pl(x+ct)

```

##### FDTD
FDTD scheme satisfies the traveling wave solution p(m,n) = pr(n−m) + pl(n+m) 
```
p(x+X,t)−2p(x,t) + p(x−X,t)/(X**2)
=1/(c**2)*(p(x,t +T)−2p(x,t) + p(x,t −T))/(T**2)
X = cT
t = nT 
x = mX
p(m,n) = p(m+1,n−1) + p(m−1,n−1)− p(m,n−2)

p(m,n) = pr(n−m) + pl(n+m)

p(m,n) = pr ((n−1)−(m+1))  + pl((n−1) + (m+1))
        + pr ((n−1)−(m−1)) + pl((n−1) + (m−1))
        − pr ((n−2)−m)     − pl((n−2) +m)
```

fs = 1/T = c/d

##### tube model
###### two end model
pR[i]|  |pL[i+1]
pR[i]= (1-r)*pR[i-1]+r*pL[i+1])
pL[i+1]= (1+r)*pL[i+1]+rpR[i-1]


## tools
### text splitter + tokenizer + g2p
### audio → VAD sentence splitter
### integrate forced aligner
### extract mel + pitch + energy features
### JSON structure + database
#### emotion classifier + visualization UI
