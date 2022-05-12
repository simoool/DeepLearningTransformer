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

def carica_file(data_dir, MAX_FILE_SIZE=100000, MAX_LENGTH=60):
    #load first language to list
    lista_italiano = []
    file_italiano = open(data_dir + '/italian-english/italian.txt','r', encoding='utf8')
    for i, (frase) in enumerate(file_italiano):
        if i < MAX_FILE_SIZE:
            lista_italiano.append(frase)
        else:
            break

    # load second langauge to list
    lista_inglese = []
    file_inglese = open(data_dir + '/italian-english/english.txt', 'r', encoding='utf8')
    for i, (frase) in enumerate(file_inglese):
        if i < MAX_FILE_SIZE:
            lista_inglese.append(frase)
        else:
            break

    # normalizza le frasi italiano e inglese
    frasi_italiano_normalizzate = list(map(normalizeString, lista_italiano))
    frasi_inglese_normalizzate = list(map(normalizeString, lista_inglese))

    frasi_italiano = []
    frasi_inglese = []

    # per tutte le frasi caricate, prende solo i primi MAX_LENGTH tokens
    for i in range(len(frasi_italiano_normalizzate)):
        tokens1 = frasi_italiano_normalizzate[i].split(' ')
        tokens2 = frasi_inglese_normalizzate[i].split(' ')
        if len(tokens1) <= MAX_LENGTH and len(tokens2) <= MAX_LENGTH:
            frasi_italiano.append(frasi_italiano_normalizzate[i])
            frasi_inglese.append(frasi_inglese_normalizzate[i])

    diz_italiano = Dictionary('italian')
    diz_inglese = Dictionary('english')
    return diz_italiano, diz_inglese, frasi_italiano, frasi_inglese

#takes in a sentence and dictionary, and tokenizes based on dictionary
def tokenize(frase, vocabolario, MAX_LENGTH=60):
    frase_spezzata = [parola for parola in frase.split(' ')]
    token = [SOS_TOKEN]
    token += [vocabolario.word2index[parola] for parola in frase.split(' ')]
    token.append(EOS_TOKEN)
    token += [PAD_TOKEN]*(MAX_LENGTH - len(frase_spezzata))
    return token

#create dataloader from a batch size and the two language lists
def load_batches(lingua_italiano, lingua_inglese, batch_size):
    data_loader = []
    for i in range(0, len(lingua_italiano), batch_size):
        seq_length = min(len(lingua_italiano) - batch_size, batch_size)
        input_batch = lingua_italiano[i:i+seq_length][:]
        target_batch = lingua_inglese[i:i+seq_length][:]
        input_tensor = torch.LongTensor(input_batch)
        target_tensor = torch.LongTensor(target_batch)
        data_loader.append([input_tensor, target_tensor])
    return data_loader