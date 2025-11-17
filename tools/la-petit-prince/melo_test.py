def split_line(input_path):
    import re
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile(r'(\n)', re.IGNORECASE)
    parts = pattern.split(content)
    all_lines=[]
    lines=""
    num_map = {
        '0': 'zéro',
        '1': 'un',
        '2': 'deux',
        '3': 'trois',
        '4': 'quatre',
        '5': 'cinq',
        '6': 'six',
        '7': 'sept',
        '8': 'huit',
        '9': 'neuf'
    }

    for i in range(0, len(parts), 2):
        body = parts[i].strip() if i < len(parts) else ""
        if body:
            for item in num_map:
                body=body.replace(item, num_map[item])
            
            all_lines.append(body)
            lines+=body
    return all_lines, lines

def openvoice():
    import os
    import torch
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    ckpt_converter = '/data/tts/OpenVoice/checkpoints_v2/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = '/data/tts/OpenVoice/outputs_v2'

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    # reference_speaker = '/data/tts/OpenVoice/resources/example_reference.mp3' # This is the voice you want to clone
    reference_speaker="/data/tts/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en_pcm.wav"
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

    os.makedirs(output_dir, exist_ok=True)

    from melo.api import TTS

    texts = {
        # 'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
        # 'EN': "Did you ever hear a folk tale about a giant turtle?",
        # 'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
        'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
        # 'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
        # 'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
        # 'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
    }


    src_path = f'{output_dir}/tmp.wav'

    # Speed is adjustable
    speed = 1.0
    # print(tone_color_converter.hps.data.sampling_rate)
    for language, text in texts.items():
        model = TTS(language=language, device=device)
        # print(model.hps.data.sampling_rate)
        from scipy.signal import resample
        
        speaker_ids = model.hps.data.spk2id
        
        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')
            
            source_se = torch.load(f'/data/tts/OpenVoice/checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
            if torch.backends.mps.is_available() and device == 'cpu':
                torch.backends.mps.is_available = lambda: False
            audio=model.tts_to_file(text, speaker_id, speed=speed)
            orig_sr = model.hps.data.sampling_rate
            target_sr = tone_color_converter.hps.data.sampling_rate
            assert target_sr  < orig_sr
            # Compute number of output samples
            num_target = int(len(audio) * target_sr / orig_sr)
            # audio = resample(audio, num_target)

            save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

            # Run the tone color converter
            encode_message = "@MyShell"
            # tone_color_converter.convert(
            tone_color_converter.convert_audio(
                # audio_src_path=src_path, 
                audio,
                src_se=source_se, 
                tgt_se=target_se, 
                # output_path=save_path,
                message=encode_message)           
def tts_model():
    import random
    import numpy as np

    from melo.api import TTS
    import soundfile
    from scipy.signal import resample

    # Speed is adjustable
    speed = 1.0

    # CPU is sufficient for real-time inference.
    # You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
    device = 'auto' # Will automatically use GPU if available
    model = TTS(language='FR', device=device)
    speaker_ids = model.hps.data.spk2id
    print(speaker_ids)
    speaker_key=next(iter(speaker_ids.keys()))
    speaker_id = speaker_ids[speaker_key]
    speaker_key = speaker_key.lower().replace('_', '-')
    # abort
    # tts.list_models()
    speakers =['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']
    text="Mon dessin ne représentait pas un chapeau. Il représentait un serpent boa qui digérait un éléphant. "
    file_path="output.wav"
    all_lines, lines=split_line("/data/tts/la-petit-prince/ch2.txt")
    all_wav=[]

    import os
    import torch
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    import librosa
    ckpt_converter = '/data/tts/OpenVoice/checkpoints_v2/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = '/data/tts/OpenVoice/outputs_v2'

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    # reference_speaker = '/data/tts/OpenVoice/resources/example_reference.mp3' # This is the voice you want to clone
    reference_speaker="/data/tts/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    # target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)
    target_se=tone_color_converter.extract_se(reference_speaker)

    for item in all_lines[:10]:
        source_se = torch.load(f'/data/tts/OpenVoice/checkpoints_v2/base_speakers/ses/fr.pth', map_location=device)
        audio=model.tts_to_file(item, speaker_id, speed=speed)
        orig_sr = model.hps.data.sampling_rate
        target_sr = tone_color_converter.hps.data.sampling_rate
        # print(orig_sr, target_sr)
        # abort
        assert target_sr  <= orig_sr
        # Compute number of output samples
        # num_target = int(len(audio) * target_sr / orig_sr)
        # audio = resample(audio, num_target)
        audio = librosa.resample(audio.T, orig_sr=orig_sr, target_sr=target_sr)
        encode_message = "@MyShell"
        wav_=tone_color_converter.convert_audio(
            audio,
            src_se=source_se, 
            tgt_se=target_se, 
            # output_path=save_path,
            message=encode_message)      

        if wav_ is not None:
            all_wav.append(wav_)
            target_sample_rate = tone_color_converter.hps.data.sampling_rate
            silence_dur = random.uniform(0.8, 1.2)
            n = int(silence_dur * target_sample_rate)
            silence_part=wav_[:n]*0
                # cross_faded_overlap = np.concatenate([prev_overlap * fade_out ,silence_part, next_overlap * fade_in])
            # cross_faded_overlap = np.concatenate([silence_part])
            all_wav.append(silence_part)
        # break
    wav = np.concatenate(all_wav)
    file_path=file_path
    # tts.synthesizer.save_wav(wav=wav, path=file_path)
    soundfile.write(file_path, wav, tone_color_converter.hps.data.sampling_rate)
    print(file_path)

   
tts_model()
# openvoice()