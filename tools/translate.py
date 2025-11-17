from transformers import MarianMTModel, MarianTokenizer

# Choose a multilingual model from Hugging Face.
# You can find supported pairs like "Helsinki-NLP/opus-mt-en-fr", "opus-mt-zh-en", etc.
model_name = "Helsinki-NLP/opus-mt-en-fr"

# model_name = "Helsinki-NLP/opus-mt-en-es"
model_name = "Helsinki-NLP/opus-mt-en-de"
model_name = "Helsinki-NLP/opus-mt-en-zh"
model_name = "Helsinki-NLP/opus-mt-zh-en"
model_name = "Helsinki-NLP/opus-mt-fr-en"
model_name = "Helsinki-NLP/opus-mt-es-en"

# Load model and tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Input text (English → French)
# src_text = ["Mon dessin ne représentait pas un chapeau. Il représentait un serpent boa qui digérait un éléphant."]
src_text = ["Cambiará el universo pero yo no, pensé con melancólica vanidad; alguna vez, lo sé, mi vana devoción la había exasperado; muerta yo podía consagrarme a su memoria, sin esperanza, pero también sin humillación."]
# src_text = ["你好,你今天好吗?"]

# Tokenize
inputs = tokenizer(src_text, return_tensors="pt", padding=True)

# Translate
translated = model.generate(**inputs)

# Decode to readable text
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

print("Source:", src_text[0])
print("Translated:", tgt_text[0])
