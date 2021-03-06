import torch

# import per ricevere eventuali argomenti di input
import argparse

# per caricamento e gestione dizionari vari
import pickle

# import da altri moduli esterni
from componenti import Transformer, Encoder, Decoder
from dictionary import Dictionary
from funzioni import *

SOS_TOKEN = 1

def carica_vocabolario(path):
    f=open(path, 'rb')
    return pickle.load(f)

def traduci_frase(frase_da_tradurre, vocab_ita, vocab_eng, model, max_len):
    
    model.eval()

    # normalizzo la frase 
    frase_normalizzata = normalizeString(frase_da_tradurre)
    
    # tokenizzazione della frase NORMALIZZATA
    frae_tokenizzata = tokenize(frase_normalizzata, vocab_ita)

    # trasformazione in tensore della frase tokenizzata
    frase_tensore = torch.LongTensor(frae_tokenizzata).unsqueeze(0)

    # calcola la input mask
    input_mask = model.make_input_mask(frase_tensore)
    
    with torch.no_grad():
        encoded_input = model.encoder(frase_tensore, input_mask)

    # token di uscita
    token_risposta = [SOS_TOKEN]

    for i in range(max_len):

        tensore_risposta = torch.LongTensor(token_risposta).unsqueeze(0)
        target_mask = model.make_target_mask(tensore_risposta)
    
        with torch.no_grad():
            output, attention = model.decoder(tensore_risposta, encoded_input, target_mask, input_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        token_risposta.append(pred_token)
        if pred_token == EOS_TOKEN:
            break
    
    # contiene lista di token della traduzione finale
    vettore_traduzione =[]
    for i in token_risposta:
        vettore_traduzione.append(vocab_eng.index2word[i])

    return ' '.join(vettore_traduzione[1:-1]), attention

def main():
    #take in arguments
    parser = argparse.ArgumentParser(description='Iperparametri per la rete')

    # parameters needed to enhance image
    parser.add_argument('--frase', type=str, default='Riprese della sessione', help='Frase da tradurre')

    #default hyperparameters 
    parser.add_argument('--MAX_LENGTH', type=int, default=60, help='numero max di tokens')
    parser.add_argument('--hidden_size', type=int, default=256, help='number of hidden layers in transformer')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=3, help='number of decoder layers')
    parser.add_argument('--encoder_heads', type=int, default=8, help='number of encoder heads')
    parser.add_argument('--decoder_heads', type=int, default=8, help='number of decoder heads')
    parser.add_argument('--encoder_ff_size', type=int, default=512, help='fully connected input size for encoder')
    parser.add_argument('--decoder_ff_size', type=int, default=512, help='fully connected input size for decoder')
    parser.add_argument('--encoder_dropout', type=float, default=0.1, help='dropout for encoder feed forward')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='dropout for decoder feed forward')

    parser.add_argument('--percorso_file', type=str, default='modelli_salvati/', help='percorso dove salvare i dati')
  
    args = parser.parse_args()

    frase_italiano = args.frase
    percorso_file = args.percorso_file
    #hyper parameters
    MAX_LENGTH = args.MAX_LENGTH
    hidden_size = args.hidden_size
    encoder_layers = args.encoder_layers
    decoder_layers = args.decoder_layers
    encoder_heads = args.encoder_heads
    decoder_heads = args.decoder_heads
    encoder_ff_size = args.encoder_ff_size
    decoder_ff_size = args.decoder_ff_size
    encoder_dropout = args.encoder_dropout
    decoder_dropout = args.decoder_dropout

    transformer_location = percorso_file + 'italiano2inglese/'

    #carico dizionari italiano e inglese
    input_lang_dic = carica_vocabolario('modelli_salvati/italiano2inglese/input_dic.pkl')
    output_lang_dic = carica_vocabolario('modelli_salvati/italiano2inglese/output_dic.pkl')

    input_size = input_lang_dic.n_count
    output_size = output_lang_dic.n_count
    
    #creo encoder e decoder
    encoder = Encoder(input_size, hidden_size, encoder_layers, encoder_heads, encoder_ff_size, encoder_dropout)
    decoder = Decoder(output_size, hidden_size, decoder_layers, decoder_heads, decoder_ff_size, decoder_dropout)

    # creo oggetto transformer
    transformer = Transformer(encoder, decoder)
    transformer.load_state_dict(torch.load(transformer_location + 'transformer_model.pt'))

    traduzione, attention = traduci_frase(frase_italiano, input_lang_dic, output_lang_dic, transformer, MAX_LENGTH)
    print("Frase italiano" + ': ' + frase_italiano)
    print("Traduzione" + ': ' + traduzione)

if __name__ == "__main__":
    main()