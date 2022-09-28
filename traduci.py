import torch

# import per ricevere eventuali argomenti di input
import argparse

# per caricamento e gestione dizionari vari
import pickle

# import da altri moduli esterni
from componenti import Transformer, Encoder, Decoder
from dictionary import Dictionary
from funzioni import *

# token che indica inizio della stringa: SOS= Start of String
SOS_TOKEN = 1

# Carico i dizionari che durante il train erano stati creati e serializzati
def carica_vocabolario(path):
    f = open(path, 'rb')
    return pickle.load(f)



def traduci_frase(frase_da_tradurre, vocab_ita, vocab_eng, model, max_len):
    
    # model.eval disattiva alcune parti che vengono usate solo per il training 
    # come per esempio Dropouts Layers, BatchNorm Layers
    model.eval()

    # Normalizzazione la frase 
    frase_normalizzata = normalizeString(frase_da_tradurre)
    
    # Tokenizzazione della frase normalizzata
    frase_tokenizzata = tokenize(frase_normalizzata, vocab_ita)

    # Trasformazione in tensore della frase tokenizzata
    frase_tensore = torch.LongTensor(frase_tokenizzata).unsqueeze(0)

    # Calcola la input mask
    input_mask = model.make_input_mask(frase_tensore)
    
    with torch.no_grad():
        encoded_input = model.encoder(frase_tensore, input_mask)

    # Token che andrà a contenere gli indici delle parole tradotte
    # il primo è sempre il token di inizio stringa: SOS
    token_risposta = [SOS_TOKEN]

    # Ciclo for per andare a riempire token_risposta, passando attraverso il decoder
    for i in range(max_len):
        tensore_risposta = torch.LongTensor(token_risposta).unsqueeze(0)
        target_mask = model.make_target_mask(tensore_risposta)
    
        with torch.no_grad():
            output, attention = model.decoder(tensore_risposta, encoded_input, target_mask, input_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        token_risposta.append(pred_token)
        
        # Interrompo quando arrivo al token EOS
        if pred_token == EOS_TOKEN:
            break
    
    # Ottengo le parole relative agli indici appena calcolati
    vettore_traduzione =[]
    for i in token_risposta:
        vettore_traduzione.append(vocab_eng.index2word[i])

    # restituisco la frase tradotta e il livello di attention
    return ' '.join(vettore_traduzione[1:-1]), attention




def main():
    # Crea l'oggetto parser di tipo ArgumentParser a cui associare gli argomenti da utilizzare
    parser = argparse.ArgumentParser(description='Iperparametri per la rete')


    parser.add_argument('--frase', type=str, default='Oggi è una bellissima giornata', help='Frase da tradurre')

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
  
    # Assegnazione vera e propria dei parametri
    args = parser.parse_args()

    frase_italiano = args.frase
    percorso_file = args.percorso_file

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

    # Carico i dizionari
    input_lang_dic = carica_vocabolario('modelli_salvati/italiano2inglese/input_dic.pkl')
    output_lang_dic = carica_vocabolario('modelli_salvati/italiano2inglese/output_dic.pkl')

    # Ottengo il numero di parole salvate dai due dizionari
    input_size = input_lang_dic.n_count
    output_size = output_lang_dic.n_count
    
    # Creo oggetti encoder e decoder
    encoder = Encoder(input_size, hidden_size, encoder_layers, encoder_heads, encoder_ff_size, encoder_dropout)
    decoder = Decoder(output_size, hidden_size, decoder_layers, decoder_heads, decoder_ff_size, decoder_dropout)

    # Creo oggetto transformer
    transformer = Transformer(encoder, decoder)
    transformer.load_state_dict(torch.load(transformer_location + 'transformer_model.pt'))

    # invocazione metodo traduci frase per avviare la traduzione della frase letta in ingresso
    traduzione, attention = traduci_frase(frase_italiano, input_lang_dic, output_lang_dic, transformer, MAX_LENGTH)
    
    # stampo risultato: FRASE ITALIANO E POI CORRISPONDENTE TRADUZIONE
    print("Frase italiano" + ': ' + frase_italiano)
    print("Traduzione" + ': ' + traduzione)



if __name__ == "__main__":
    main()