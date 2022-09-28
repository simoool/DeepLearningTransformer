import re
import unicodedata
import torch

from dictionary import Dictionary

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2




#https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )




def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s




# Carica i dataset (normalizzandoli e controllando la lunghezza delle frasi) e crea gli oggetti Dizionario
def carica_file(data_dir, MAX_FILE_SIZE=100000, MAX_LENGTH=60):
    # Apre il dataset in italiano e lo salva frase per frase in 'lista_italiano', fino a un numero di frasi pari a MAX_FILE_SIZE
    lista_italiano = []
    file_italiano = open(data_dir + '/italian-english/italian.txt','r', encoding='utf8')
    for i, (frase) in enumerate(file_italiano):
        if i < MAX_FILE_SIZE:
            lista_italiano.append(frase)
        else:
            break

    # Apre il dataset in inglese e lo salva frase per frase in 'lista_inglese', fino a un numero di frasi pari a MAX_FILE_SIZE
    lista_inglese = []
    file_inglese = open(data_dir + '/italian-english/english.txt', 'r', encoding='utf8')
    for i, (frase) in enumerate(file_inglese):
        if i < MAX_FILE_SIZE:
            lista_inglese.append(frase)
        else:
            break

    # Normalizza le frasi italiano e inglese
    frasi_italiano_normalizzate = list(map(normalizeString, lista_italiano))
    frasi_inglese_normalizzate = list(map(normalizeString, lista_inglese))

    frasi_italiano = []
    frasi_inglese = []

    # Aggiunge alle nuove liste 'frasi_italiano' e 'frasi_inglese' solo le frasi che hanno lunghezza (in termini di parole) < MAX_LENGTH.
    # Per farlo scorre le frasi precedentemente salvate, e per ognuna crea una lista con le singole parole della frase stessa (es: ['ripresa', 'della', 'sessione']). Quindi controlla la lunghezza della lista corrente e, se è < MAX_LENGTH, la salva. 
    for i in range(len(frasi_italiano_normalizzate)):
        tokens1 = frasi_italiano_normalizzate[i].split(' ')
        tokens2 = frasi_inglese_normalizzate[i].split(' ')
        if len(tokens1) <= MAX_LENGTH and len(tokens2) <= MAX_LENGTH:
            frasi_italiano.append(frasi_italiano_normalizzate[i])
            frasi_inglese.append(frasi_inglese_normalizzate[i])

    # Crea gli oggetti dizionario
    diz_italiano = Dictionary('italiano')
    diz_inglese = Dictionary('inglese')

    return diz_italiano, diz_inglese, frasi_italiano, frasi_inglese




# Riceve una frase e il dizionario, quindi tokenizza la frase aggiungendo anche i token SOS, EOS e PAD
# Ogni token è rappresentato dall'indice nel dizionario della relativa parola
def tokenize(frase, vocabolario, MAX_LENGTH=60):
    frase_spezzata = [parola for parola in frase.split(' ')]
    token = [SOS_TOKEN]
    token += [vocabolario.word2index[parola] for parola in frase.split(' ')]
    token.append(EOS_TOKEN)
    token += [PAD_TOKEN]*(MAX_LENGTH - len(frase_spezzata))
    return token




# Partendo dalle liste di frasi tokenizzate, costruisce delle batch prendendone batch_size a ogni iterazione.
# Successivamente, partendo dalle batch, costruisce i rispettivi tensori. Questi verranno restituiti nella lista (a coppie) data_loader
def load_batches(lingua_italiano, lingua_inglese, batch_size):
    data_loader = []
    for i in range(0, len(lingua_italiano), batch_size):
        input_batch = lingua_italiano[i:i+batch_size][:]
        target_batch = lingua_inglese[i:i+batch_size][:]
        input_tensor = torch.LongTensor(input_batch)
        target_tensor = torch.LongTensor(target_batch)
        data_loader.append([input_tensor, target_tensor])
    return data_loader