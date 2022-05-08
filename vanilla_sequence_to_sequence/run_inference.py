import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

def run_inferencing(model, enc_layers, dec_layers, type_, units):
    # get the output of the model from the last encoder layer after training 
    if(type_ == 'RNN'):
        enc_op, state = model.layers[enc_layers + 3].output
        enc_state = [state]
    elif(type_ == 'LSTM'):
        enc_op, enc_h, enc_c = model.layers[enc_layers + 3].output
        enc_state = [enc_h, enc_c]
    elif(type_ == 'GRU'):
        enc_op, state = model.layers[enc_layers + 3].output
        enc_state = [state]
    # initializing to save all the hidden states and input layers 
    dec_ip_st = []
    dec_op_st = []
    enc_ip = model.input[0]
    enc_model = keras.Model(enc_ip, enc_state)
    dec_ip = keras.Input(shape=( 1))
    final_= None
    if(type_ == 'RNN'):
        for x in range(dec_layers):
            # supplying the hidden state through the input layer
            dec_st_ip = keras.Input(shape = (units,))
            z = [dec_st_ip]
            dec_rnn = model.layers[x + enc_layers + 4]
            # retrieving the decoder layers and passing the ouput to the next layer
            if (x == 0):
                dec_op, st = dec_rnn(model.layers[x + enc_layers + 2](dec_ip), initial_state=z)
            else:
                dec_op, st = dec_rnn(final_, initial_state=z)
            final_ = dec_op 
            dec_ip_st.append(dec_st_ip)
            dec_op_st.append(st)
    elif(type_ == 'LSTM'):
        for x in range(dec_layers):
            # supplying the hidden state and the cell state through the input layer
            dec_st_ip_h = keras.Input(shape = (units,))
            dec_st_ip_c = keras.Input(shape = (units,))
            z = [dec_st_ip_h, dec_st_ip_c]
            dec_lstm = model.layers[x + enc_layers + 4]
            # retrieving the decoder layers and passing the ouput to the next layer
            if (x == 0):
                dec_op, dec_st_h, dec_st_c = dec_lstm(model.layers[x + enc_layers + 2](dec_ip), initial_state=z)
            else:
                dec_op, dec_st_h, dec_st_c = dec_lstm(final_, initial_state=z)
            final_ = dec_op
            dec_ip_st.append(dec_st_ip_h)
            dec_ip_st.append(dec_st_ip_c)
            dec_op_st.append(dec_st_h)
            dec_op_st.append(dec_st_c)
    elif(type_ == 'GRU'):
        for x in range(dec_layers):
            # supplying the hidden state and the cell state through the input layer
            dec_st_ip = keras.Input(shape = (units,))
            z = [dec_st_ip]
            dec_gru = model.layers[x + enc_layers + 4]
            # retrieving the decoder layers and passing the ouput to the next layer
            if (x == 0):
                dec_op, st = dec_gru(model.layers[x + enc_layers + 2](dec_ip), initial_state=z)
            else:
                dec_op, st = dec_gru(final_, initial_state=z)
            final_ = dec_op
            dec_ip_st.append(dec_st_ip)
            dec_op_st.append(st)
    dense_layer = model.get_layer('last_layer')
    dec_op = dense_layer(final_)
    final_model = keras.Model([dec_ip]+ dec_ip_st, [dec_op] + dec_op_st)
    return enc_model, final_model 