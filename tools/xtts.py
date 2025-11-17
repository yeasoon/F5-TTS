def tts_model():
    from TTS.api import TTS

    # List available models
    # TTS.list_models()  

    # Initialize model
    # tts = TTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=True, gpu=False)
    # tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)
    # tts.list_models()
    speakers =['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']
    # Generate speech
    # tts.tts_to_file(text="Hola, esta es una lectura más humana de tu texto.",
    #                 file_path="output_human.wav")

    tts.tts_to_file(
    # text="O God, I could be bounded in a nutshell and count myself a King of infinite space, Hamlet, II, 2",
    # text="But they will teach us that Eternity is the Standing still of the Present Time, a Nunc-stans (as the Schools call it); which neither they, nor any else understand, no more than they would a Hic-stans for an Infinite greatness of Place. Leviathan, IV, 46",
    # text="La candente mañana de febrero en que Beatriz Viterbo murió, después de una imperiosa agonía que no se rebajó un solo instante ni al sentimentalismo ni al miedo, ",
    # text="noté que las carteleras de fierro de la Plaza Constitución habían renovado no sé qué aviso de cigarrillos rubios;",
    # text="el hecho me dolió, pues comprendí que el incesante y vasto universo ya se apartaba de ella y que ese cambio era el primero de una serie infinita.",
    text="Cambiará el universo pero yo no, pensé con melancólica vanidad; alguna vez, lo sé, mi vana devoción la había exasperado; muerta yo podía consagrarme a su memoria, sin esperanza, pero también sin humillación.",
    # text="至若松竹含韵，梧楸蚤脱，惊绮疏之晓吹，堕碧砌之凉月。念塞外之征行，顾闺中之骚屑。夜蛩鸣兮机杼促，朔雁叫兮音书绝。远件续兮何冷冷，虚窗静兮空切切。如吟如啸，非竹非丝。合自然之宫微，动终岁之别离。废井苔冷，荒园露滋。草苍苍兮人寂寂，树槭槭兮虫咿咿。则有安石风流，巨源多可，平六符而佐主，施九流而自我。犹复感阴虫之鸣轩，叹凉叶之初堕。异宋玉之悲伤，觉潘郎之幺麽",
    # text="そういう時、己は、向うの山の頂の巖いわに上り、空谷くうこくに向って吼ほえる。この胸を灼く悲しみを誰かに訴えたいのだ。",
    speaker=speakers[9],  # built-in speaker
    language="es",
    file_path="output.wav"
)
tts_model()