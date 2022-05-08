from pickletools import optimize
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from make_predictions import predict

from data_prepare import PrepareData
from model_ import create_model
from run_inference import run_inferencing
from make_predictions import predict
from calc_accuracy import calculate_accuracy
import wandb
from wandb.keras import WandbCallback


# preparing all the train, test, and validation data and data tokenizers for input and output data
dpp = PrepareData()
train_ip_text, train_ip_data, train_ip_tokenizer, train_op_text, train_op_data, train_op_tokenizer = dpp.prepare_data('dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv', 1)
val_ip_text, val_ip_data, val_ip_tokenizer, val_op_text, val_op_data, val_op_tokenizer = dpp.prepare_data('dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv', 0)
test_ip_text, test_ip_data, test_ip_tokenizer, test_op_text, test_op_data, test_op_tokenizer = dpp.prepare_data('dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv', 0)

encoder_tokens = len(train_ip_tokenizer.word_index)+1
decoder_tokens = len(train_op_tokenizer.word_index)+1
max_enc_len =  train_ip_data.shape[1]
max_dec_len = train_op_data.shape[1]

# creating a dictionary for the index of characters
id_char_ip = dict((train_ip_tokenizer.word_index[key], key) for key in train_ip_tokenizer.word_index.keys())
id_char_op = dict((train_op_tokenizer.word_index[key], key) for key in train_op_tokenizer.word_index.keys())

def train():
    wandb.init()
    type_ = str(wandb.config.type_)
    rnn_units = int(wandb.config.units)
    enc_layers = int(wandb.config.enc_layers)
    dec_layers = int(wandb.config.dec_layers)
    embed_dim = int(wandb.config.embed_dim)
    batch_size = int(wandb.config.batch_size)
    drpout = int(wandb.config.dropout)
    epochs = int(wandb.config.epochs)

    wandb.run.name = 'type_' + type_ + '_units_' + str(rnn_units) + '_enc_lrs_' + str(enc_layers) + '_dec_lrs_' + str(dec_layers) + '_embed_dim_' + str(embed_dim) + '_bs_' + str(batch_size) + '_dropout_' + str(drpout) + '_epochs_' + str(epochs)

    # creates the model
    model = create_model(type_, embed_dim, enc_layers, dec_layers, drpout, max_enc_len, max_dec_len, encoder_tokens, decoder_tokens, rnn_units)
    # compiles the model with the hyperparameters provided
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(reduction='none'), metrics=['acc'])

    for x in range(epochs):
        # start training the model
        model_obj = model.fit([train_ip_data, train_op_data], tf.concat([train_op_data[:,1:],tf.zeros((train_op_data.shape[0],1))], axis=1), batch_size=batch_size, epochs=epochs, shuffle=True, callbacks=[WandbCallback(save_model=(False))])

        # run the inference model
        enc , dec = run_inferencing(model, enc_layers, dec_layers, type_, rnn_units)
        
        # calculate the train and validation accuracy and log it to wandb
        wandb.log({"train_loss" : model_obj.history['loss'][0]})
        val_acc = calculate_accuracy(enc, dec, enc_layers, dec_layers, val_ip_data, val_op_text, 0, val_ip_text, type_, train_op_tokenizer, max_dec_len, id_char_op)
        wandb.log({"val_acc":val_acc})
    print('Test accuracy : ', calculate_accuracy(enc, dec, enc_layers, dec_layers, test_ip_data, test_op_text, 1, test_ip_text, type_, train_op_tokenizer, max_dec_len, id_char_op))

train()
    


