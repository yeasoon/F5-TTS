from transformers import MarianMTModel, MarianTokenizer
import re
model_name = "Helsinki-NLP/opus-mt-fr-en"
# Load model and tokenizer
tokenizer0 = MarianTokenizer.from_pretrained(model_name)
model0 = MarianMTModel.from_pretrained(model_name)
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer1 = MarianTokenizer.from_pretrained(model_name)
model1 = MarianMTModel.from_pretrained(model_name)

def split_line(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile(r'(\n)', re.IGNORECASE)
    parts = pattern.split(content)
    all_lines=[]
    lines=""
    for i in range(0, len(parts), 2):
        body = parts[i].strip() if i < len(parts) else ""
        if body:
            all_lines.append(body)
            lines+=body
    return all_lines, lines

def fr2enzh(text):
    # Input text (English → French)
    # src_text = ["Mon dessin ne représentait pas un chapeau. Il représentait un serpent boa qui digérait un éléphant."]

    # src_text = ["你好,你今天好吗?"]

    # Tokenize
    inputs = tokenizer0(text, return_tensors="pt", padding=True)

    # Translate
    translated = model0.generate(**inputs)

    # Decode to readable text
    tgt_text0 = tokenizer0.batch_decode(translated, skip_special_tokens=True)

    # Tokenize
    inputs = tokenizer1(tgt_text0, return_tensors="pt", padding=True)

    # Translate
    translated = model1.generate(**inputs)

    # Decode to readable text
    tgt_text1 = tokenizer1.batch_decode(translated, skip_special_tokens=True)
    for idx, item in enumerate(text):
        print("fr:", item)
        print("en:", tgt_text0[idx])
        print("zh:", tgt_text1[idx])
        print()

# src_text = ["Mon dessin ne représentait pas un chapeau. Il représentait un serpent boa qui digérait un éléphant."]
all_lines, lines=split_line("/data/tts/la-petit-prince/ch1.txt")
print(all_lines[0])
fr2enzh(all_lines)