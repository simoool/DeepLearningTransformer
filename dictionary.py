PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2


class Dictionary:

    # Inizializza l'oggetto dizionario
    def __init__(self, name):
        self.name = name
        self.word2index = {}    # Indica l'indice di ciascuna parola --> es: {'ripresa': 3, 'della': 4, 'sessione': 5, ...}
        self.word2count = {}    # Indica quante volte compare ogni parola --> es: {'ripresa': 1, 'della': 1, 'sessione': 1, ...}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"} # Aggiunge le parole nuove --> es: {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'ripresa', 4: 'della', 5: 'sessione', ...}
        self.n_count = 3    # Tiene il conto del numero di parole diverse


    # Riceve una frase e la spezza, chiamando per ogni parola la funzione add_word
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


    # Riceve una parola e di conseguenza aggiorna le informazioni del dizionario
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_count
            self.word2count[word] = 1
            self.index2word[self.n_count] = word
            self.n_count += 1
        else:
            self.word2count[word] += 1
