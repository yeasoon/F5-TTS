meta_info="/data/tts/la-petit-prince/all"
def get_set(meta_info):
    text_vocab_set = set()
    with open(meta_info, "r") as f:
        lines = f.readlines()
        for line in lines:
            # norm_text = line.strip()
            norm_text = line
            # norm_text.replace(" ","")
            text_vocab_set.update(list(norm_text))
    return text_vocab_set
dict0=get_set(meta_info)
print(dict0)
dict1=get_set("/data/tts/F5-TTS/src/f5_tts/infer/examples/vocab.txt")
# print(dict1)
for item in dict0:
    if item not in dict1:
        print(item)
