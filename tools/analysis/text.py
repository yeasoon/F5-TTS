# normalizer.py
import re
def normalize_text(text: str) -> str:
    punctuation_sbmls={
    "ascii": ["!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~"],
    "chinese": ["，", "。", "、", "？", "！", "：", "；", "——", "……", "“", "”", "‘", "’", "（", "）", "【", "】", "《", "》", "〈", "〉", "『", "』", "「", "」", "·", "￥"],
    "math": ["+", "-", "×", "÷", "=", "≠", "≈", "∞", "√", "∑", "∏", "∫", "∂", "∇", "±", "≤", "≥", "°", "%"],
    "arrows": ["→", "←", "↑", "↓", "↔", "⇒", "⇐", "⇑", "⇓"],
    "currency": ["¥", "$", "€", "£", "₩", "₹"]
    }

    text = text.strip()
    text = re.sub(r'\s+', ' ', text)              # collapse spaces
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)
    text = re.sub(r'[!！]', '!', text)
    text = re.sub(r'[?？]', '?', text)
    text = re.sub(r'[(（]', '(', text)
    text = re.sub(r'[)）]', ')', text)
    text = re.sub(r'[。]', '.', text)
    text = re.sub(r'[——……]', '.', text)
    text = re.sub(r'[：；]', ',', text)
    text = re.sub(r'[,，]', ',', text)
    
    return text

# splitter.py
def split_paragraphs(text: str):
    return [p.strip() for p in text.split("\n") if p.strip()]

def split_sentences(paragraph: str):
    sentences = re.split(r'([.!?,])', paragraph)
    return sentences

# g2p.py
from phonemizer import phonemize

def g2p_en(text: str):
    ph = phonemize(text, language="en-us", backend="espeak", strip=True, punctuation_marks=';:,.!?')
    return ph.split()

def char_tokenize(text: str):
    return list(text)

from pypinyin import pinyin, Style
from pypinyin import load_phrases_dict, load_single_dict
load_phrases_dict({'古朴': [['gu3'], ['pu3']]})
load_phrases_dict({'老长': [['lao3'], ['chang3']]})
load_phrases_dict({'弹指': [['tan2'], ['zhi3']]})
load_phrases_dict({'倒栽': [['dao4'], ['zai1']]})
load_phrases_dict({'何处': [['he2'], ['chu4']]})
load_phrases_dict({'未干': [['wei4'], ['gan1']]})
load_phrases_dict({'重来': [['chong2'], ['lai2']]})
load_phrases_dict({'仙乐': [['xian1'], ['yue4']]})
load_phrases_dict({'长教': [['zhang3'], ['jiao4']]})
load_phrases_dict({'明了': [['ming2'], ['liao3']]})
load_phrases_dict({'见长': [['jian4'], ['zhang3']]})
load_phrases_dict({'宝相': [['bao3'], ['xiang4']]})
load_phrases_dict({'还生': [['huan2'], ['sheng']]})
load_single_dict({ord('切'): 'qie1,qie4'}) 
def g2p_zh(text: str):
    # load_single_dict({ord('还'): 'hái,huán'}) 
    py = pinyin(text, style=Style.TONE3, errors='ignore', heteronym=True)
    return [p[0] for p in py]

from g2pM import G2pM
g2p = G2pM()
def g2p_z_v1(text: str):
    return g2p(text)

import stanza
# stanza.download('zh')
nlp_zh = stanza.Pipeline('zh')


def split_word(text: str,lang="en"):

    doc = nlp_zh(text)
    return [w.text for s in doc.sentences for w in s.words]

def process_text(raw: str, lang="en"):
    text = normalize_text(raw)

    paragraphs = split_paragraphs(text)
    results = []
    # print(paragraphs)
    for p in paragraphs:
        sentences = split_sentences(p)
        # print(sentences)
        for s in sentences:
            words = split_word(s, lang=lang)
            chars = char_tokenize(s)

            if lang == "en":
                phonemes = g2p_en(s)
            else:
                phonemes = g2p_zh(s)

            results.append({
                "paragraph": p,
                "sentence": s,
                "words": words,
                "chars": chars,
                "phonemes": phonemes
            })
    return results
s="我还记得先前的医生的议论和方药，和现在所知道的比较起来，便渐渐的悟得中医不过是一种有意的或无意的骗子"
print(process_text(s, lang="zh"))
# print([item for item in g2p_z_v1(s) if item not in ['，','。']])
# print(g2p_zh(s))
# s="what i am dreaming is to get a terminal peace"
# print(g2p_en(s))
