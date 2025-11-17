
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


def generate(model, speakers, text, file_path, speaker_id=9, language="fr"):
    wav = model.tts(
    # text="Mon dessin ne représentait pas un chapeau. Il représentait un serpent boa qui digérait un éléphant. ",
    text=text,
    # speaker=speakers[speaker_id],  # built-in speaker
    # language="fr",
    speed=1.0,
    )
    return wav
   
def tts_model():
    from TTS.api import TTS
    import random
    import numpy as np

    # List available models
    # TTS.list_models()  

    # Initialize model
    # tts = TTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=True, gpu=False)
    # tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)
    tts = TTS(model_name="tts_models/fr/css10/vits", progress_bar=True, gpu=False)
    # # tts = TTS(model_name="tts_models/fr/mai/tacotron2-DDC", progress_bar=True, gpu=False)
    # # tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=True, gpu=False)
    
    
    # tts.list_models()
    speakers =['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']
    text="Mon dessin ne représentait pas un chapeau. Il représentait un serpent boa qui digérait un éléphant. "
    file_path="output1.wav"
    all_lines, lines=split_line("/data/tts/la-petit-prince/ch2.txt")
    all_wav=[]
    for item in all_lines:
        wav_=generate(tts, speakers, item, file_path, speaker_id=11, language="fr")
        if wav_:
            all_wav.append(wav_)

            target_sample_rate = 24000
            silence_dur = random.uniform(0.8, 1.2)
            n = int(silence_dur * target_sample_rate)
            silence_part=wav_[:n]*0
                # cross_faded_overlap = np.concatenate([prev_overlap * fade_out ,silence_part, next_overlap * fade_in])
            # cross_faded_overlap = np.concatenate([silence_part])
            all_wav.append(silence_part)
        # break
    wav = np.concatenate(all_wav)
    file_path=file_path
    tts.synthesizer.save_wav(wav=wav, path=file_path)
    print(file_path)

   
tts_model()