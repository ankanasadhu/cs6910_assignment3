from turtle import shape
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

def create_model(type_, embed_dim, enc_layers, dec_layers, dropout, max_enc_len, max_dec_len, num_enc_tok, num_dec_tok, units):
    # both the encoder and decoder takes in the tokenized inputs
    enc_ip = keras.Input(shape=( max_enc_len))
    dec_ip = keras.Input(shape=( max_dec_len))
    # the embedding layer creates the vector embeddings for the inputs
    embedding = keras.layers.Embedding(num_enc_tok, embed_dim)(enc_ip)
    dec_embedding = keras.layers.Embedding(num_dec_tok, embed_dim)(dec_ip)
    # we store the final encoding of 1 layer
    final_enc=None
    # create the encoder for RNN
    if(type_ == 'RNN'):
        # if there is only one encoding layer, just return the states 
        if(enc_layers == 1):
            enc = keras.layers.SimpleRNN(units, return_state=True, dropout=dropout)
            enc_op, op = enc(embedding)
        else:
            for x in range(enc_layers - 1):
                enc = keras.layers.SimpleRNN(units, return_sequences=True, dropout=dropout)
                # send the embedding to the first layer
                if (x == 0):
                    enc_op = enc(embedding)
                # otherwise send the previous encoding to the next encoding layer
                else:
                    enc_op = enc(final_enc)
                final_enc = enc_op
            # for the last encoder layer we return the states
            enc = keras.layers.SimpleRNN(units, return_state=True, dropout=dropout)
            enc_op, op = enc(final_enc)
        enc_states = [op]
        for x in range(dec_layers):
            # get both the sequence and the current state from the decoder
            dec_rnn = keras.layers.SimpleRNN(units, return_sequences=True, return_state=True, dropout=dropout)
            # send the embeddings to the first decoding layer
            if(x == 0):
                dec_op, _ = dec_rnn(dec_embedding, initial_state=enc_states)
            # otherwise send the previous decoding to the next layer
            else:
                dec_op, _ = dec_rnn(final, initial_state=enc_states)
            final = dec_op
        # add the dense layer
        dec_dense = keras.layers.Dense(num_dec_tok, activation='Softmax', name='last_layer')
        # pass the final decoding to the dense layer
        dec_op = dec_dense(final)
    # create the encoder for LSTM
    elif(type_ == 'LSTM'):
        # if there is only one encoding layer, just return the states 
        if(enc_layers == 1):
            enc = keras.layers.LSTM(units, return_state=True, dropout=dropout)
            enc_op, op_h, op_c = enc(embedding)
        else:
            # adding all the layers except the last one because the last layer returns the states
            for x in range(enc_layers - 1):
                enc = keras.layers.LSTM(units, return_sequences=True, dropout=dropout)
                # send the embedding to the first layer
                if(x == 0):
                    enc_op = enc(embedding)
                # otherwise send the previous encoding to the next encoding layer
                else:
                    enc_op = enc(final_enc)
                final_enc = enc_op
            # for the last encoder layer we return the states
            enc = keras.layers.LSTM(units, return_state=True, dropout=dropout)
            enc_op, op_h, op_c = enc(final_enc)
        # encoder states are hidden and cell states for the last layer
        enc_states = [op_h, op_c]
        for x in range(dec_layers):
            #  get both the sequence and the current state from the decoder
            dec_lstm = keras.layers.LSTM(units, return_sequences=True, return_state=True, dropout=dropout)
            # send the embeddings to the first decoding layer
            if(x == 0):
                dec_op, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)
            # otherwise send the previous decoding to the next layer
            else:
                dec_op, _, _ = dec_lstm(final, initial_state=enc_states)
            final = dec_op
        # add the dense layer
        dec_dense = keras.layers.Dense(num_dec_tok, activation='Softmax', name='last_layer')
        dec_op = dec_dense(final)

    elif(type_ == 'GRU'):
        # if there is only one encoding layer, just return the states 
        if(enc_layers == 1):
            enc = keras.layers.GRU(units, return_state=True, dropout=dropout)
            enc_op, op = enc(embedding)
        else:
            # adding all the layers except the last one because the last layer returns the states
            for x in range(enc_layers - 1):
                enc = keras.layers.GRU(units, return_sequences=True, dropout=dropout)
                if (x == 0):
                    enc_op = enc(embedding)
                else:
                    enc_op = enc(final_enc)
                final_enc = enc_op
            # for the last encoder layer we return the states
            enc = keras.layers.GRU(units, return_state=True, dropout=dropout)
            enc_op, op = enc(final_enc)
        # encoder state is the hidden state for the last layer
        enc_states = [op]
        for x in range(dec_layers):
            #  get both the sequence and the current state from the decoder
            dec_gru = keras.layers.GRU(units, return_sequences=True, return_state=True, dropout=dropout)
            if(x == 0):
                # send the embeddings to the first decoding layer
                dec_op, _ = dec_gru(dec_embedding, initial_state=enc_states)
            else:
                # otherwise send the previous decoding to the next layer
                dec_op, _ = dec_gru(final, initial_state=enc_states)
            final = dec_op
        # add the dense layer
        dec_dense = keras.layers.Dense(num_dec_tok, activation='Softmax', name='last_layer')
        dec_op = dec_dense(final)
    
    # make the model from the encoder inputs, inputs to the decoder for teacher forcing and the decoder output
    model = keras.Model([enc_ip, dec_ip], dec_op)
    return model




