# import librerie esterne

import torch
import time
import argparse
import torch.optim as optim
from random import shuffle
import pickle

from componenti import *
from funzioni import *

# Costanti usate anche negli altri moduli
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Trainer():

    # Inizializza i pesi con una distribuzione uniforme di tipo xavier_uniform, controllando che l'oggetto transformer abbia l'attributo "weight" di dimensione > 1.
    def inizializza_pesi(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.xavier_uniform_(model.weight.data)

    # inizializzazione pesi avviene con una xavier uniform, funzione fornita da torch.
    # inizializza i pesi con valori assunti tra -a ed a, dove a è calcolato come segue:
    # a= gain* 6/((fan_in+fan_out)^1/2)

    # Inizializzazione dell'oggetto transformer
    def __init__(self,data_directory, MAX_LENGTH, MAX_FILE_SIZE, batch_size, lr=0.0005, hidden_size=256, encoder_layers=3, decoder_layers=3,
                 encoder_heads=8, decoder_heads=8, encoder_ff_size=512, decoder_ff_size=512, encoder_dropout=0.1, decoder_dropout=0.1):

        self.MAX_LENGTH = MAX_LENGTH
        self.MAX_FILE_SIZE = MAX_FILE_SIZE

        # Assegnazione degli oggetti dizionario e delle liste con le frasi prese dai dataset
        self.dizionario_ita, self.dizionario_ing, self.lista_frasi_ita, self.lista_frasi_ing = carica_file(data_directory, self.MAX_FILE_SIZE, self.MAX_LENGTH)
        
        # Aggiunge le frasi prese dai dataset ai rispettivi dizionari
        for frase in self.lista_frasi_ita:
            self.dizionario_ita.add_sentence(frase)
        for frase in self.lista_frasi_ing:
            self.dizionario_ing.add_sentence(frase)

        # Serializza entrambi i dizionari
        # Serializzazione realizzata con pickle e comando dump
        # highest protocol è valore costante, inserito obbligatoriamente nel comando dump

        with open('modelli_salvati/' + self.dizionario_ita.name + '2' + self.dizionario_ing.name + '/input_dic.pkl', 'wb') as f:
            pickle.dump(self.dizionario_ita, f, pickle.HIGHEST_PROTOCOL)
        with open('modelli_salvati/' + self.dizionario_ita.name + '2' + self.dizionario_ing.name + '/output_dic.pkl', 'wb') as f:
            pickle.dump(self.dizionario_ing, f, pickle.HIGHEST_PROTOCOL)

        # Assegnazione della lista di frasi tokenizzate per entrambi dataset
        self.frasi_ita_token = [tokenize(sentence, self.dizionario_ita, self.MAX_LENGTH) for sentence in self.lista_frasi_ita]
        self.frasi_ing_token = [tokenize(sentence, self.dizionario_ing, self.MAX_LENGTH) for sentence in self.lista_frasi_ing]

        self.batch_size = batch_size

        # Assegnazione della lista contenente le coppie di tensori, costruiti partendo dalle frasi tokenizzate
        self.data_loader = load_batches(self.frasi_ita_token, self.frasi_ing_token, self.batch_size)

        # Numero di parole diverse in ognuno dei due dizionari
        input_size = self.dizionario_ita.n_count
        output_size = self.dizionario_ing.n_count

        # Creazione degli oggetti Encoder e Decoder
        encoder_part = Encoder(input_size, hidden_size, encoder_layers, encoder_heads, encoder_ff_size, encoder_dropout)
        decoder_part = Decoder(output_size, hidden_size, decoder_layers, decoder_heads, decoder_ff_size, decoder_dropout)

        # Creazione dell'oggetto transformer, di cui poi vengono inizializzati i pesi
        self.transformer = Transformer(encoder_part, decoder_part, PAD_TOKEN)
        self.transformer.apply(self.inizializza_pesi)

        # Definizione della funzione di loss e dell'optimizer, utilizzando rispettivamente Cross-Entropy e Adam
        self.loss_func = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=lr)


    def train(self, epochs, saved_model_directory):
        start_time = time.time()

        for epoch in range(epochs):

            # Mescola le batch per prevenire l'overfitting
            shuffle(self.data_loader)

            # determina il momento di inizio delle operazioni
            start_time = time.time()

            # definizione train loss, inizialmente=0
            train_loss = 0

            for input, target in self.data_loader:

                # Imposta il gradiente a zero, perchè a ogni backpropagation vengono si somma il gradiente corrente con i precedenti
                self.optimizer.zero_grad()

                # Ottiene l'output e la sua dimensione passando attraverso il transformer
                output, _ = self.transformer(input, target[:,:-1])
                output_dim = output.shape[-1]

                # Esegue una flatten di output e di target, togliendo da quest'ultimo il token SOS
                # view(-1, output_dim) --> reshape del tensore con n° righe indefinito, n° colonne = output_dim
                # view(-1) --> reshape del tensore su un unica riga
                output = output.contiguous().view(-1, output_dim)
                target = target[:,1:].contiguous().view(-1)

                # Calcola la loss in base ai risultati ottenuti
                loss = self.loss_func(output, target)

                # Backpropagation e aggiornamento dei parametri
                loss.backward()
                nn.utils.clip_grad_norm_(self.transformer.parameters(), 1) # Gradient clipping, per evitare il problema dell'exploding gradient
                self.optimizer.step()

                # Calcola la loss per la batch
                train_loss += loss.item()
                
            # Calcola la loss per l'epoca
            train_loss /= len(self.data_loader)

            end_time = int(time.time() - start_time)
            torch.save(self.transformer.state_dict(), saved_model_directory + self.dizionario_ita.name +
            '2' + self.dizionario_ing.name + '/transformer_model_{}.pt'.format(epoch))

            # stampo info utili
            print('Epoca: {}  -->  Tempo trascorso: {}s  -  Tempo stimato rimanente: {}s.'.format(epoch, end_time, (epochs-epoch)*end_time))
            print('\tLoss: {:.4f}\n'.format(train_loss))
        print('Train terminato!')


def main():

    # Crea l'oggetto parser di tipo ArgumentParser a cui associare gli argomenti da utilizzare
    parser = argparse.ArgumentParser(description='Hyperparameters for training Transformer')

    # Aggiunta degli argomenti (iperparametri) e delle directory
    parser.add_argument('--MAX_LENGTH', type=int, default=60, help='max number of tokens in input')
    parser.add_argument('--MAX_FILE_SIZE', type=int, default=1000, help='max number of lines to read from files') #100000
    parser.add_argument('--batch_size', type=int, default=128, help='size of batches passed through networks at each step')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate of models')
    parser.add_argument('--hidden_size', type=int, default=256, help='number of hidden layers in transformer')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=3, help='number of decoder layers')
    parser.add_argument('--encoder_heads', type=int, default=8, help='number of encoder heads')
    parser.add_argument('--decoder_heads', type=int, default=8, help='number of decoder heads')
    parser.add_argument('--encoder_ff_size', type=int, default=512, help='fully connected input size for encoder')
    parser.add_argument('--decoder_ff_size', type=int, default=512, help='fully connected input size for decoder')
    parser.add_argument('--encoder_dropout', type=float, default=0.1, help='dropout for encoder feed forward')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='dropout for decoder feed forward')
    parser.add_argument('--epoche', type=int, default=10, help='number of iterations of dataset through network for training')
    
    parser.add_argument('--data_directory', type=str, default='data', help='data directory')
    parser.add_argument('--saved_model_directory', type=str, default='modelli_salvati/', help='data directory')

    # Assegnazione vera e propria dei parametri
    args = parser.parse_args()

    # Creazione di nuove variabili contenenti gli oggetti passati per argomento
    # Epoche e percorso verranno passati per il train, tutti gli altri per l'inizializzazione
    data_directory = args.data_directory
    MAX_LENGTH = args.MAX_LENGTH
    MAX_FILE_SIZE = args.MAX_FILE_SIZE
    batch_size = args.batch_size
    lr = args.lr
    hidden_size = args.hidden_size
    encoder_layers = args.encoder_layers
    decoder_layers = args.decoder_layers
    encoder_heads = args.encoder_heads
    decoder_heads = args.decoder_heads
    encoder_ff_size = args.encoder_ff_size
    decoder_ff_size = args.decoder_ff_size
    encoder_dropout = args.encoder_dropout
    decoder_dropout = args.decoder_dropout
    epoche = args.epoche
    percorso = args.saved_model_directory

    # Creazione dell'oggetto transformer (come oggetto di Trainer, dove poi verrà creato il transformer vero e proprio)
    transformer = Trainer(data_directory, MAX_LENGTH, MAX_FILE_SIZE, batch_size, lr, hidden_size, encoder_layers, decoder_layers, 
                            encoder_heads, decoder_heads, encoder_ff_size, decoder_ff_size, encoder_dropout, decoder_dropout)

    # Viene chiamato il metodo train sul transformer appena creato e inizializzato
    # verrà avviata la fase di training
    transformer.train(epoche, percorso)


if __name__ == "__main__":
    main()
