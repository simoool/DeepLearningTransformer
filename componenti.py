# Modulo che contiene tutti le "componenti fondamentali" che costituiscono il Transformer

import torch 
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hidden_size, n_heads, dropout):
        super().__init__()
        
        assert hidden_size % n_heads == 0
        
        # Assegnazione parametri
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        
        # Creazione di Q,V,K,out come funzioni lineari
        self.fc_query = nn.Linear(hidden_size, hidden_size)
        self.fc_key = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
    
        self.dp = nn.Dropout(dropout)
        
        self.coefficient = torch.sqrt(torch.FloatTensor([self.head_size]))
        

    def forward(self, query, key, value, mask=None):
        b_size = query.shape[0]
   
        # Assegnazione dei tensori Q,V,K
        query_output = self.fc_query(query)
        key_output = self.fc_key(key)
        value_output = self.fc_value(value)

        # Riformulazione delle dimensioni matriciali di Q,V,K
        query_output = query_output.view(b_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        key_output = key_output.view(b_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        value_output = value_output.view(b_size, -1, self.n_heads, self.head_size).permute(0, 2, 1, 3)
      
        # (Q*K)/sqrt(head_size)
        score = torch.matmul(query_output, key_output.permute(0, 1, 3, 2)) / self.coefficient
        if mask is not None:
            # Metto dei numeri molto piccoli in corrispondenza delle posizioni in cui avrei i PAD token, così che non vengano considerate 
            score = score.masked_fill(mask == 0, -1e10)
        
        # Softmax
        attention = torch.softmax(score, dim = -1)    
        # Moltiplico v per l'ultimo risultato ottenuto, ottenendo il risultato della Scaled Dot-Product Attention
        output = torch.matmul(self.dp(attention), value_output)
        # Eseguo la concatenazione dei prodotti ottenuti, faccio passare il risultato attraverso un layer lineare e restituisco l'output della Multi-Head Attention
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(b_size, -1, self.hidden_size)  
        output = self.fc_out(output)

        return output




class FeedForwardLayer(nn.Module):

    def __init__(self, hidden_size, ff_size, dropout):
        super().__init__()

        # Creazione del FF layer formato da un linear,una reLu, un dropout e un'altra linear, come espresso dal paper di descrizione della struttura del Transformer
        self.ff_layer = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),           
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size)
        )
        

    def forward(self, input):
        output = self.ff_layer(input)
        return output




class EncoderLayer(nn.Module):

    # Inizializzazione dei layer interni all'Encoder
    def __init__(self, hidden_size, n_heads, ff_size,  dropout):
        super().__init__()
        
        self.self_atten = MultiHeadAttentionLayer(hidden_size, n_heads, dropout)
        self.self_atten_norm = nn.LayerNorm(hidden_size)
        self.ff_layer = FeedForwardLayer(hidden_size, ff_size, dropout)
        self.dp = nn.Dropout(dropout)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        

    def forward(self, input, input_mask):

        # Attention
        atten_result = self.self_atten(input, input, input, input_mask)
        # Normalizzazione
        atten_norm = self.self_atten_norm(input + self.dp(atten_result))
        # Feed-Forward layer
        ff_result = self.ff_layer(atten_norm)
        # Normalizzazione
        output = self.ff_layer_norm(atten_norm + self.dp(ff_result))

        return output




class Encoder(nn.Module):

    # Inizializzazione dei layer di encoding
    def __init__(self, input_size, hidden_size, n_layers, n_heads, ff_size,dropout, MAX_LENGTH=100):
        super().__init__()
        
        self.te = nn.Embedding(input_size, hidden_size)
        self.pe = nn.Embedding(MAX_LENGTH, hidden_size)
        
        # Creo la sequenza di layer di encoding, tanti quanti forniti come iperparametro
        encoding_layers = []
        for _ in range(n_layers):
            encoding_layers.append(EncoderLayer(hidden_size, n_heads, ff_size, dropout))
        self.encode_sequence = nn.Sequential(*encoding_layers)
        
        self.dp = nn.Dropout(dropout)
        
        self.coefficient = torch.sqrt(torch.FloatTensor([hidden_size]))
        

    def forward(self, input, input_mask):
        b_size = input.shape[0]
        input_size = input.shape[1]
        
        # Embedding + Positional Encoding dell'input
        pos = torch.arange(0, input_size).unsqueeze(0).repeat(b_size, 1)
        input = self.dp((self.te(input) * self.coefficient) + self.pe(pos))

        # Passaggio per gli n_layers di encoding
        for layer in self.encode_sequence:
            input = layer(input, input_mask)
  
        return input




class DecoderLayer(nn.Module):

    # Inizializzazione dei layer interni al Decoder
    def __init__(self, hidden_size, n_heads, ff_size, dropout):
        super().__init__()
        
        self.self_atten = MultiHeadAttentionLayer(hidden_size, n_heads, dropout)
        self.self_atten_norm = nn.LayerNorm(hidden_size)
        self.encoder_atten = MultiHeadAttentionLayer(hidden_size, n_heads, dropout)
        self.encoder_atten_norm = nn.LayerNorm(hidden_size)
        self.ff_layer = FeedForwardLayer(hidden_size, ff_size, dropout)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        self.dp = nn.Dropout(dropout)
        

    def forward(self, target, encoded_input, target_mask, input_mask):
        # Attention
        atten_result = self.self_atten(target, target, target, target_mask)
        # Normalizzazione
        atten_norm = self.self_atten_norm(target + self.dp(atten_result))
        # Attention
        atten_encoded = self.encoder_atten(atten_norm, encoded_input, encoded_input, input_mask)
        # Normalizzazione
        encoded_norm = self.encoder_atten_norm(atten_norm + self.dp(atten_encoded))
        # Feed-Forward layer
        ff_result = self.ff_layer(encoded_norm)
        # Normalizzazione
        output = self.ff_layer_norm(encoded_norm + self.dp(ff_result))

        return output




class Decoder(nn.Module):

    # Inizializzazione dei layer di encoding
    def __init__(self, output_size, hidden_size, n_layers, n_heads, ff_size, dropout, MAX_LENGTH=100):
        super().__init__()
        
        self.te = nn.Embedding(output_size, hidden_size)
        self.pe = nn.Embedding(MAX_LENGTH, hidden_size)

        decoding_layers = []
        for _ in range(n_layers):
            decoding_layers.append(DecoderLayer(hidden_size, n_heads, ff_size, dropout))
        
        self.decode_sequence = nn.Sequential(*decoding_layers) 
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.dp = nn.Dropout(dropout)
        
        self.coefficient = torch.sqrt(torch.FloatTensor([hidden_size]))
        

    def forward(self, target, encoded_input, target_mask, input_mask):    
        b_size = target.shape[0]
        target_size = target.shape[1]
        
        # Embedding + Positional Encoding dell'input (tensore contenente i token relativi alle parole in inglese, che si aggiorna a ogni passaggio per il decoder)
        pos = torch.arange(0, target_size).unsqueeze(0).repeat(b_size, 1)
        target = self.dp((self.te(target) * self.coefficient) + self.pe(pos))

        # Passaggio per gli n_layers di decoding
        for layer in self.decode_sequence:
            target = layer(target, encoded_input, target_mask, input_mask)

        output = self.fc_out(target)

        return output




class Transformer(nn.Module):

    def __init__(self, encoder, decoder, padding_index=0):
        super().__init__()
        
        # Assegnazione dei parametri passati in input, tra cui le strutture Encoder e Decoder
        self.encoder = encoder
        self.decoder = decoder
        self.padding_index = padding_index
        

    # Funzione per la creazione della maschera di input
    def make_input_mask(self, input):
        input_mask = (input != self.padding_index).unsqueeze(1).unsqueeze(2)
        return input_mask
    

    # Funzione per la creazione della maschera target
    def make_target_mask(self, target):
        target_pad_mask = (target != self.padding_index).unsqueeze(1).unsqueeze(2)
        target_sub_mask = torch.tril(torch.ones((target.shape[1], target.shape[1]))).bool()
        target_mask = target_pad_mask & target_sub_mask
        return target_mask


    def forward(self, input, target):
        # Creazione maschere di input e target  
        input_mask = self.make_input_mask(input)
        target_mask = self.make_target_mask(target)

        # Chiamata all'Encoder
        encoded_input = self.encoder(input, input_mask)

        # Chiamata al Decoder
        output = self.decoder(target, encoded_input, target_mask, input_mask) 

        return output  