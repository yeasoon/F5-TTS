def ja_enc(text=None):
    import pykakasi

    # Create a kakasi converter
    kks = pykakasi.kakasi()

    text = "私は東京に行きます "

    # Convert text (new API)
    result = kks.convert(text)

    # Each element is a dict like {'orig': '私', 'hira': 'わたし', 'hepburn': 'watashi'}
    romaji = " ".join([item["hepburn"] for item in result])

    print(romaji)  # watashi wa toukyou ni ikimasu
ja_enc()
# # jp_to_pinyin.py
# from opencc import OpenCC
# from pypinyin import pinyin, Style

# cc = OpenCC('jp2t')  # attempt: map Japanese kanji -> Chinese variant (if available)
# text_jp = "私は東京に行きます。"
# text_cn = cc.convert(text_jp)   # may map to same or similar characters

# # convert Chinese chars to pinyin (approximate Mandarin reading)
# py = pinyin(text_cn, style=Style.TONE3, heteronym=False)
# py_flat = [item[0] for item in py]
# print("japanese:", text_jp)
# print("mapped-chinese:", text_cn)
# print("pinyin:", " ".join(py_flat))
