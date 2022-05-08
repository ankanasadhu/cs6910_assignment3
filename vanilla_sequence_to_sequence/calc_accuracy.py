import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from make_predictions import predict

# this function gets all the predicted words and writes both the right and wrong predictions on separate files
# format : input_romanized_word  corresponding_target_word predicted_word  
def calculate_accuracy(enc_model, dec_model, enc_layers, dec_layers, ip_, op_text, test_, ip_text, type_, op_tokenizer, max_dec_len, id_char_op):
    count = 0
    bs = ip_.shape[0]
    predictions = predict(ip_, enc_model, dec_model, bs , enc_layers, dec_layers, type_, op_tokenizer, max_dec_len, id_char_op)
    for x in range(bs):
        pred_ = predictions[x]
        t_word = op_text[x][1:-1]
        if(t_word == pred_):
            count += 1
            if(test_):
                file_ = open("predictions_vanilla/correct.txt", 'a')
                file_.write(ip_text[x] + ' ' + t_word + ' ' + pred_ + '\n')
                file_.close()
        else:
            if(test_):
                file_ = open("predictions_vanilla/wrong.txt", 'a')
                file_.write(ip_text[x] + ' ' + t_word + ' ' + pred_ + '\n')
                file_.close()
    return float(count) / float(bs)
