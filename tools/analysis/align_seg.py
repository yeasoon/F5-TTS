def align(audio, text,output_dir=None):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    inference_pipline = pipeline(
        task=Tasks.speech_timestamp,
        model='iic/speech_timestamp_prediction-v1-16k-offline',
        # output_dir=output_dir,
    )
    rec_result = inference_pipline(input=(audio, text), data_type=("sound", "text"))
    return rec_result
def split_audio(audio, text):
    from pydub import AudioSegment
    ret=align(audio, text)[0]
    audio = AudioSegment.from_wav(audio)
    for i, seg in enumerate(ret["timestamp"]):
        start_ms = int(seg[0])
        end_ms = int(seg[1])

        chunk = audio[0:1000]   # slice
        chunk.export(f"/data/tts/F5-TTS/tools/analysis/tmp/segment_{i}.wav", format="wav")
        break
ret=split_audio("/data/tts/F5-TTS/dataset/zh/wavs/call_to_arms/call_to_arms_0009.wav", "然而我的父亲终于日重一日的亡故了。有谁从小康人家而坠入困顿的么")
print(ret)
