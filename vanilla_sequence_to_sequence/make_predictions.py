import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

def predict(ip_, enc_model, dec_model, bs, enc_layer, dec_layer, type_, op_tokenizer, max_dec_len, id_char_op):
    # get all encoder ouput
    enc_op = enc_model.predict(ip_)
    if(type_ == 'RNN' or 'GRU'):
        enc_op = [enc_op]
    temp = enc_op
    for _ in range(dec_layer - 1):
        temp = temp +  enc_op
    enc_op = temp

    # store the previously predicted character's index 
    char_ind = np.zeros((bs, 1))
    # all words start with '\t'
    char_ind[:, 0] = op_tokenizer.word_index['\t']

    pred = ["" for _ in range(bs)]
    bools = [0 for _ in range(bs)]

    for x in range(max_dec_len):
        op = dec_model.predict(tuple([char_ind] + enc_op))
        prob = op[0]
        enc_op = op[1:]
        for y in range(bs):
            if bools[y]:
                continue
            # find the index of the most likely character
            tok_ind = np.argmax(prob[y, -1, :])
            if tok_ind == 0:
                char = '\n'
            else:
                char = id_char_op[tok_ind]
            if char == '\n':
                bools[y] = 1
                continue
            pred[y] += char
            # update the predicted characters
            char_ind[y, 0] = op_tokenizer.word_index[char]
    return pred

