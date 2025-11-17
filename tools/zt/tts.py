import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode
import torch
from f5_tts.infer.utils_infer import (
    cfg_strength,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    sway_sampling_coef,
    save_spectrogram,
)
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
from f5_tts.model import DiT, UNetT

import re

def init(config):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "xpu"
        if torch.xpu.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # config file

    # config = tomli.load(open(args.config, "rb"))


    # command-line interface parameters
    # -----------------------------------------
    # target_sample_rate = 24000
    # n_mel_channels = 100
    # hop_length = 256
    # win_length = 1024
    # n_fft = 1024
    # mel_spec_type = "vocos"
    # target_rms = 0.1
    # cross_fade_duration = 0.15
    # ode_method = "euler"
    # nfe_step = 32  # 16, 32
    # cfg_strength = 2.0
    # sway_sampling_coef = 0.0
    # speed = 1.2
    # fix_duration = None
    target_sample_rate = 24000
    n_mel_channels = 100
    hop_length = 256
    win_length = 1024
    n_fft = 1024
    mel_spec_type = "vocos"
    target_rms = 0.1
    cross_fade_duration = 0.15
    ode_method = "euler"
    nfe_step = 32  # 16, 32
    cfg_strength = 2.0
    sway_sampling_coef = -1.0
    speed = 1.0
    fix_duration = None
    # -----------------------------------------


    model =  config.get("model", "F5TTS_v1_Base")
    ckpt_file =config.get("ckpt_file", "")
    vocab_file = config.get("vocab_file", "")

    ref_audio =  config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
    ref_text =config.get("ref_text", "Some call me nature, others call me mother nature.")

    output_dir = config.get("output_dir", "tests")
    output_file = config.get(
        "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
    )

    save_chunk = config.get("save_chunk", False)
    use_legacy_text = config.get("no_legacy_text", False)  # no_legacy_text is a store_false arg
    if save_chunk and use_legacy_text:
        print(
            "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
        )

    remove_silence = config.get("remove_silence", False)
    load_vocoder_from_local = config.get("load_vocoder_from_local", False)

    vocoder_name =  config.get("vocoder_name", mel_spec_type)
    target_rms = config.get("target_rms", target_rms)
    cross_fade_duration = config.get("cross_fade_duration", cross_fade_duration)
    nfe_step =  config.get("nfe_step", nfe_step)
    cfg_strength = config.get("cfg_strength", cfg_strength)
    sway_sampling_coef = config.get("sway_sampling_coef", sway_sampling_coef)
    speed = config.get("speed", speed)
    fix_duration =  config.get("fix_duration", fix_duration)
    device = config.get("device", device)


    # patches for pip pkg user
    if "infer/examples/" in ref_audio:
        ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
   
    if "voices" in config:
        for voice in config["voices"]:
            voice_ref_audio = config["voices"][voice]["ref_audio"]
            if "infer/examples/" in voice_ref_audio:
                config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))


    # ignore gen_text if gen_file provided


    # output path

    wave_path = Path(output_dir) / output_file
    # spectrogram_path = Path(output_dir) / "infer_cli_out.png"
    if save_chunk:
        output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
        if not os.path.exists(output_chunk_dir):
            os.makedirs(output_chunk_dir)


    # load vocoder

    if vocoder_name == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    elif vocoder_name == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

    vocoder = load_vocoder(
        vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
    )


    # load TTS model

    model_cfg = OmegaConf.load(
        config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
    )
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

    if model != "F5TTS_Base":
        assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

    # override for previous models
    if model == "F5TTS_Base":
        if vocoder_name == "vocos":
            ckpt_step = 1200000
        elif vocoder_name == "bigvgan":
            model = "F5TTS_Base_bigvgan"
            ckpt_type = "pt"
    elif model == "E2TTS_Base":
        repo_name = "E2-TTS"
        ckpt_step = 1200000

    if not ckpt_file:
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))
        # ckpt_file = str(cached_path(f"hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/model_last_reduced.pt"))
    # abort
    # vocab_file = str(cached_path(f"hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/vocab.txt"))

    print(f"Using {model}...")
    # ema_model = load_model(
    #     DiT, dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,text_mask_padding=False,pe_attn_head=1), ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device
    # )
    # ckpt_file="/data/tts/F5-TTS/ckpts/LJSpeech_v0/model_last.pt"
    # vocab_file="/data/tts/F5-TTS/data/LJSpeech_pinyin/vocab.txt"
    # ckpt_file="/data/tts/F5-TTS/ckpts/lesmis1/model_last.pt"
    # vocab_file="/data/tts/F5-TTS/data/lesmis1_char/vocab.txt"
    # ckpt_file="/data/tts/F5-TTS/ckpts/lesmis/model_last.pt"
    # vocab_file="/data/tts/F5-TTS/data/lesmis_char/vocab.txt"
    ckpt_file="/data/tts/F5-TTS/ckpts/ru_0/model_last.pt"
    vocab_file="/data/tts/F5-TTS/data/ru_char/vocab.txt"
    # ckpt_file="/data/tts/F5-TTS/ckpts/zh/model_last.pt"
    # vocab_file="/data/tts/F5-TTS/data/zh_pinyin/vocab.txt"/data/tts/F5-TTS/ckpts/en_fr
    # ckpt_file="/data/tts/F5-TTS/ckpts/en_fr/model_last.pt"
    # vocab_file="/data/tts/F5-TTS/data/en_fr_char/vocab.txt"
    # print(model_arc)
    # abort
    ema_model = load_model(
        model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device
    )

    return (
    ref_audio, 
    ref_text, 
    ema_model, 
    vocoder,
    target_sample_rate,
    n_mel_channels,
    hop_length,
    win_length,
    n_fft,
    vocoder_name,
    target_rms,
    cross_fade_duration,
    ode_method,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    device,
    )

def generate(config, gen_text,
    ref_audio, 
    ref_text, 
    ema_model, 
    vocoder,
    target_sample_rate,
    n_mel_channels,
    hop_length,
    win_length,
    n_fft,
    vocoder_name,
    target_rms,
    cross_fade_duration,
    ode_method,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    device,
    gen_file="",
    lang=None,):
    if "infer/examples/" in gen_file:
        gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
    if gen_file:
        gen_text = codecs.open(gen_file, "r", "utf-8").read()

    # inference process
    if True:
        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
        if "voices" not in config:
            voices = {"main": main_voice}
        else:
            voices = config["voices"]
            voices["main"] = main_voice
        for voice in voices:
            # print("Voice:", voice)
            # print("ref_audio ", voices[voice]["ref_audio"])
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
            # print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, gen_text)
        reg2 = r"\[(\w+)\]"
        for text in chunks:
            if not text.strip():
                continue
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                # print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                # print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            ref_audio_ = voices[voice]["ref_audio"]
            ref_text_ = voices[voice]["ref_text"]
            local_speed = voices[voice].get("speed", speed)
            gen_text_ = text.strip()
            # print(f"Voice: {voice}")
            # print(ref_text_)
            # print(local_speed)
            # abort
            # local_speed=1.2
            audio_segment, final_sample_rate, spectrogram = infer_process(
                ref_audio_,
                ref_text_,
                gen_text_,
                ema_model,
                vocoder,
                mel_spec_type=vocoder_name,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=local_speed,
                fix_duration=fix_duration,
                device=device,
                lang=lang,
            )
            save_spectrogram(spectrogram, "tests/spectrogram.jpg")
            generated_audio_segments.append(audio_segment)

            # if save_chunk:
            #     if len(gen_text_) > 200:
            #         gen_text_ = gen_text_[:200] + " ... "
            #     if use_legacy_text:
            #         gen_text_ = unidecode(gen_text_)
            #     sf.write(
            #         os.path.join(output_chunk_dir, f"{len(generated_audio_segments) - 1}_{gen_text_}.wav"),
            #         audio_segment,
            #         final_sample_rate,
            #     )

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)

            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            return final_wave, final_sample_rate
            # with open(wave_path, "wb") as f:
            #     sf.write(f.name, final_wave, final_sample_rate)
            #     # Remove silence
            #     if remove_silence:
            #         remove_silence_for_generated_wav(f.name)
            #     print(f.name)
        else:
            return None, None

def keep_chinese_and_punctuation(text):
    # Keep only: Chinese characters (\u4e00-\u9fff)
    # and common punctuation marks (。，“”，！？；：《》、“”…)
    pattern = re.compile(r'[^\u4e00-\u9fff。，、！？；：“”‘’（）《》〈〉——……—·「」『』]')
    ret=pattern.sub('', text)
    ret = ret.replace("曰", "日")
    ret = ret.replace("朴", "普")
    ret = ret.replace("弹指", "谈指")
    ret = ret.replace("倒栽", "到栽")
    ret = ret.replace("刺刺", "旯旯")
    ret = ret.replace("傻十三", "傻逼")
    ret = ret.replace("何处", "何触")
    ret = ret.replace("……", "，")
    ret = ret.replace("未干", "未甘")
    ret = re.sub(r' +', ' ', ret)
    ret = re.sub(r'([。，、！？；：“”‘’（）《》〈〉——……—·「」『』])。', r'\1', ret)
    ret = re.sub(r'([。，、！？；：“”‘’（）《》〈〉——……—·「」『』])，', r'\1', ret)
   
    # return re.sub(pattern, '。', ret)
    return ret

def split_line(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile(r'(\n)', re.IGNORECASE)
    parts = pattern.split(content)
    all_lines=[]
    lines=""
    body=""
    for i in range(1, len(parts), 2):
      
        body0 = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # body0 = keep_chinese_and_punctuation(body0)
        # body = body0
        if body0:
            body=body+" "+ body0+"。"
        if len(body)<10 and i < len(parts)-2:
            continue
        body = keep_chinese_and_punctuation(body)
        all_lines.append(body)
        lines+=body
        body=""
        # print(body)
    # print(lines)
    # abort
    return all_lines, lines

def gen_line(txt_idx=1):
    all_lines, lines=split_line(f"/data/tts/zt/split/{txt_idx}.txt")
    # audio_wave, sr=generate()
    config={
    "model":"F5TTS_v1_Base",
    # "ref_audio": "/data/tts/zt/ref.wav",
    # "ref_text":"生命是世间最伟大的奇迹，星空中的青铜巨棺。",
    # "ref_audio": "/data/tts/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav",
    # "ref_text":"Some call me nature, others call me mother nature",
    # "ref_audio": "/data/tts/zt/ref_fr.wav",
    # "ref_text":"Mon dessin ne représentait pas un chapeau. Il représentait un serpent boa qui digérait un éléphant. ",
    # "ref_audio": "/data/tts/F5-TTS/dataset/LJSpeech/LJSpeech-1.1/wavs/LJ001-0001.wav",
    # "ref_text":"Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
    # "ref_audio": "/data/tts/F5-TTS/dataset/lesmis/wavs/lesmis/lesmis_0001.wav",
    # "ref_text":"Chapitre I La Charybde du faubourg Saint-Antoine et la Scylla du faubourg du Temple",
    "ref_audio": "/data/tts/F5-TTS/dataset/ru/wavs/early_short_stories/early_short_stories_0001.wav",
    "ref_text":"За столицей мудрого царя Соломона шелестел по склонам холмов густой лес. С его опушки запутанные тропинки вели на поляну,",
    # "ref_audio": "/data/tts/F5-TTS/dataset/zh_new/part_all/sentence_0_1.wav",
    # "ref_text":"今天我想和你聊聊我们所处的这个时代。",
    
    "remove_silence": False,
    "output_dir":"tests",
    "output_file":"infer_cli_basic_new.wav",
    "speed":1.0,

    }
    # lang=["fr"]
    lang=None
    
    # generate(config, text, gen_file="/data/tts/zt/split/1.txt")
    generated_audio_segments = []
    (ref_audio, 
        ref_text, 
        ema_model, 
        vocoder,
        target_sample_rate,
        n_mel_channels,
        hop_length,
        win_length,
        n_fft,
        vocoder_name,
        target_rms,
        cross_fade_duration,
        ode_method,
        nfe_step,
        cfg_strength,
        sway_sampling_coef,
        speed,
        fix_duration,
        device) = init(config)
    for idx, text in enumerate(all_lines[:1]):
    # if True:
        # text="生命是世间最伟大的奇迹，星空中的青铜巨棺。"
        # text="Lorsque j’avais six ans j’ai vu, une fois, une magnifique image, dans un livre sur la Forêt Vierge qui s’appelait « Histoires Vécues »."
        # text="Sir, in my heart there was a kind of fighting That would not let me sleep. Methought I lay Worse than the mutines in the bilboes."
        # text="Moi non plus. Il alluma une cigarette à l’aide d’une allumette-bougie qu’il agita plusieurs fois pour l’éteindre."
        text="девицу Тамару, жилище которой лежит за пределами Иерусалима. И я хочу знать, правду ли говорит Авиноам?"
        # if idx < 18:
        #     continue
        # final_text_list = convert_char_to_pinyin([text])
        # for idx, item in enumerate(text):
        #     print(idx, item)
        # text=lines
        print(text)
        # final_text_list = convert_char_to_pinyin([text])
        # final_text_list = convert_char_to_pinyin(["日"])
        # print(final_text_list)
        audio_segment, sr = generate(config, text,
        ref_audio, 
        ref_text, 
        ema_model, 
        vocoder,
        target_sample_rate,
        n_mel_channels,
        hop_length,
        win_length,
        n_fft,
        vocoder_name,
        target_rms,
        cross_fade_duration,
        ode_method,
        nfe_step,
        cfg_strength,
        sway_sampling_coef,
        speed,
        fix_duration,
        device,
        lang=lang,
        )
        if audio_segment is not None:
            generated_audio_segments.append(audio_segment)
        # break
    # if False:
    if True:
        final_wave = np.concatenate(generated_audio_segments)
    else:
        final_wave = generated_audio_segments[0]
        cross_fade_samples = int(cross_fade_duration * target_sample_rate)
        cross_fade_samples = min(cross_fade_samples, len(final_wave))
        fade_in = np.linspace(0, 1, cross_fade_samples)
        final_wave[:cross_fade_samples] = final_wave[:cross_fade_samples]*fade_in
        for i in range(1, len(generated_audio_segments)):
            prev_wave = final_wave
            next_wave = generated_audio_segments[i]
            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            import random
            # n = random.randint(20, 100)
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue
            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]
            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)
            # Cross-faded overlap
            # print(prev_overlap.shape, prev_overlap.dtype)
            silence_dur = random.uniform(0.8, 1.2)
            # silence_dur = random.uniform(4.8, 5.2)
            n = int(silence_dur * target_sample_rate)
            silence_part=prev_wave[:n]*0
            # cross_faded_overlap = np.concatenate([prev_overlap * fade_out ,silence_part, next_overlap * fade_in])
            cross_faded_overlap = np.concatenate([silence_part])
            # Combine
            new_wave = np.concatenate(
                # [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                 [prev_wave[:], cross_faded_overlap, next_wave[:]]
            )
            final_wave = new_wave
        # fade_out = np.linspace(1, 0, 100)
        # final_wave[-cross_fade_samples:] = final_wave[-cross_fade_samples:]*fade_out
    # wave_path=f"tests/infer_cli_basic_new_{txt_idx}.wav"
    wave_path=f"tests/infer_cli_basic_new.wav"
    with open(wave_path, "wb") as f:
        sf.write(f.name, final_wave, sr)
        # Remove silence
        # if config.get("remove_silence",False):
        # if True:
        #     remove_silence_for_generated_wav(f.name)
        print(f.name)
    # print(audio_wave.shape, sr)

for i in range(100,200):
    gen_line(i)
    break
