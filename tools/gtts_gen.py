def gtts():
    from gtts import gTTS

    # Your Spanish text
    text = """Mon dessin ne représentait pas un chapeau. Il représentait un serpent boa qui digérait un éléphant."""

    # Generate audio
    tts = gTTS(text=text, lang='fr')
    tts.save("output0.wav")
gtts()