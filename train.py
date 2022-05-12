# import librerie esterne

import torch
import time
import argparse
import torch.optim as optim
from random import shuffle
import pickle

from componenti import *
from funzioni import *

# costanti usate anche negli altri moduli
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Trainer():

    def inizializza_pesi(self, model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.xavier_uniform_(model.weight.data)

    def salva_dizionario(self, dizionario, input=True):
        if input is True:
            with open('modelli_salvati/' + self.input_lang_dic.name + '2' + self.output_lang_dic.name + '/input_dic.pkl', 'wb') as f:
                pickle.dump(dizionario, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('modelli_salvati/' + self.input_lang_dic.name + '2' + self.output_lang_dic.name + '/output_dic.pkl', 'wb') as f:
                pickle.dump(dizionario, f, pickle.HIGHEST_PROTOCOL)

    def __init__(self,data_directory, MAX_LENGTH, MAX_FILE_SIZE, batch_size, lr=0.0005, hidden_size=256, encoder_layers=3, decoder_layers=3,
                 encoder_heads=8, decoder_heads=8, encoder_ff_size=512, decoder_ff_size=512, encoder_dropout=0.1, decoder_dropout=0.1):

        self.MAX_LENGTH = MAX_LENGTH
        self.MAX_FILE_SIZE = MAX_FILE_SIZE

        self.input_lang_dic, self.output_lang_dic, self.input_lang_list, self.output_lang_list = carica_file(data_directory, self.MAX_FILE_SIZE, self.MAX_LENGTH)
        
        for frase in self.input_lang_list:
            self.input_lang_dic.add_sentence(frase)

        for frase in self.output_lang_list:
            self.output_lang_dic.add_sentence(frase)

        self.salva_dizionario(self.input_lang_dic, input=True)
        self.salva_dizionario(self.output_lang_dic, input=False)

        self.tokenized_input_lang = [tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH) for sentence in self.input_lang_list]
        self.tokenized_output_lang = [tokenize(sentence, self.output_lang_dic, self.MAX_LENGTH) for sentence in self.output_lang_list]

        self.batch_size = batch_size

        self.data_loader = load_batches(self.tokenized_input_lang, self.tokenized_output_lang, self.batch_size)

        input_size = self.input_lang_dic.n_count
        output_size = self.output_lang_dic.n_count

        #define encoder and decoder parts of transformer
        encoder_part = Encoder(input_size, hidden_size, encoder_layers, encoder_heads, encoder_ff_size, encoder_dropout)
        decoder_part = Decoder(output_size, hidden_size, decoder_layers, decoder_heads, decoder_ff_size, decoder_dropout)

        self.transformer = Transformer(encoder_part, decoder_part, PAD_TOKEN)
        self.transformer.apply(self.initialize_weights)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=lr)


    def train(self, epochs, saved_model_directory):
        start_time = time.time()

        for epoch in range(epochs):
            #shuffle batches to prevent overfitting
            shuffle(self.data_loader)

            start_time = time.time()
            train_loss = 0

            for input, target in self.data_loader:
                #zero gradient
                self.optimizer.zero_grad()

                #pass through transformer
                output, _ = self.transformer(input, target[:,:-1])
                output_dim = output.shape[-1]

                #flatten and omit SOS from target
                output = output.contiguous().view(-1, output_dim)
                target = target[:,1:].contiguous().view(-1)

                #loss
                loss = self.loss_func(output, target)

                #backprop
                loss.backward()
                nn.utils.clip_grad_norm_(self.transformer.parameters(), 1)
                self.optimizer.step()

                train_loss += loss.item()
                
            train_loss /= len(self.data_loader)

            end_time = int(time.time() - start_time)
            torch.save(self.transformer.state_dict(), saved_model_directory + self.input_lang_dic.name +
            '2' + self.output_lang_dic.name + '/transformer_model_{}.pt'.format(epoch))

            print('Epoch: {},   Time: {}s,  Estimated {} seconds remaining.'.format(epoch, end_time, (epochs-epoch)*end_time))
            print('\tTraining Loss: {:.4f}\n'.format(train_loss))
        print('Training finished!')

def main():
    parser = argparse.ArgumentParser(description='Hyperparameters for training Transformer')
    #hyperparameter loading
    parser.add_argument('--data_directory', type=str, default='data', help='data directory')

    #default hyperparameters (dont need to be inputed when called script)
    parser.add_argument('--MAX_LENGTH', type=int, default=60, help='max number of tokens in input')
    parser.add_argument('--MAX_FILE_SIZE', type=int, default=100000, help='max number of lines to read from files')
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

    parser.add_argument('--saved_model_directory', type=str, default='modelli_salvati/', help='data directory')
    args = parser.parse_args()

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

    transformer = Trainer(data_directory, MAX_LENGTH, MAX_FILE_SIZE, batch_size, lr, hidden_size, encoder_layers, decoder_layers, 
                            encoder_heads, decoder_heads, encoder_ff_size, decoder_ff_size, encoder_dropout, decoder_dropout)
    transformer.train(epoche, percorso)


if __name__ == "__main__":
    main()
