# Transformers for NLP: Traduzione italiano - inglese
Il progetto realizza e implementa un Transformer, rete neurale particolarmente utilizzata per diverse finalità nel mondo dell'NLP (Natural Language Processing).
Il Transformer è stato poi impiegato come traduttore, ossia come modello di Machine Learning capace, ricevuta una frase in input in italiano dall'utente, di produrre la corrispondente traduzione in inglese come output finale.


La soluzione presentata è stata sviluppata facendo riferimento a molteplici fonti e spunti di studio: 
1) **Attention Is All You Need**: https://arxiv.org/pdf/1706.03762.pdf (paper ufficiale)
2) **Transformers**: https://towardsdatascience.com/transformers-89034557de14
3) **How do Transformers Work in NLP?**:
   https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
4) **The Illustrated Transformer**: https://jalammar.github.io/illustrated-transformer/

In aggiunta, è stato consultato parzialmente il libro:  
**Transformers for Natural Language Processing: Build, train, and fine-tune deep neural network architectures for NLP with Python, PyTorch, TensorFlow, BERT, and GPT-3, 2nd Edition**  

# Dataset
Per questo progetto è stato usato il dataset relativo alla sessione parlamentare europea:"European Parliament Proceedings Parallel Corpus 1996-2011".

# Transformers: Struttura e Modello
Il Transformer è composto principalmente da due componenti: l'**Encoder** e il **Decoder**, collegati come mostrato in figura:

![](immagini/Struttura.png)

A completare l'intera struttura sono presenti un **Input Embedding** combinato con un **Positional Embedding**, così da preparare i dati ricevuti in input per le elaborazioni che verranno svolte dall'Encoder, e un **Output Embedding** che, sempre unito a un **Positional Embedding**, elabora i dati risultanti dall'Encoder per renderli utilizzabili dal Decoder.
Infine, al termine delle elaborazioni, vengono realizzate alcune attività di "post processing" che consistono in una **Softmax** e in una **Linear**.

# Organizzazione del Progetto

Il progetto è strutturato in moduli, ciascuno dei quali comprende funzionalità e componenti tra loro dipendenti e strettamente legati.
Nello specifico, vi sono 5 moduli distinti:
1) **dictionary.py**: realizza l'oggetto Dictionary per entrambe le lingue. Memorizza ogni parola e si occupa di memorizzare il numero di volte che queste compaiono nel dataset.
2) **componenti.py**: contiene tutte le componenti che vengono richieste nella struttura del Transformer: dall'Encoder al Decoder, fino alle loro rispettive componenti più dettagliate come MultiHeadAttention, FeedForwardLayer. Infine contiene la componente Transformer, ovvero l'oggetto dato dall'unione di tutte le precedenti parti.
3) **traduci.py**: modulo invocato quando si vuole eseguire una traduzione di una frase: ricevuti i parametri di input necessari, si occupa di far elaborare la frase dal Transformer per poterne ottenere la traduzione e stamparla a video.
4) **train.py**: Elabora i dataset ed esegue l'intera attività di train, così da creare le strutture dati con cui eseguire la traduzione.
5) **funzioni.py**: modulo che contiene alcune funzioni di utilità varia, come quelle per il caricamento dei dataset o per la normalizzazione delle frasi.

# Componenti e moduli python utilizzati:

Nel corso dello sviluppo delle varie componenti si è fatto uso di alcune librerie esterne, in grado di fornire supporti e funzionalità essenziali per la progettazione del Transformer:

1) numpy
2) pytorch
3) argparse
4) pickle

# Alcuni risultati ottenuti: 

**italiano**: "Io sono uno studente"  
**inglese**: "I am a student"  

**italiano**: "Roma è la capitale dell'Italia"  
**inglese**: "Rome is the capital of Italy"

**italiano**: "Oggi è una bellissima giornata"  
**inglese**: "Today is a magnificent day"  