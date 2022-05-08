import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd


# this class is used to create the dataset for training, validation and testing
# the input and target words are appended with start and end sequence 
class PrepareData:
    def __init__(self) -> None:
        self.ip_tokenizer = None
        self.op_tokenizer = None
        self.ip_len = None
        self.op_len = None

    def prepare_data(self, path, train):
        # reading the inputs from the csv file
        dataframe = pd.read_csv(path, sep='\t', names=['1','2','3']).astype(str)
        self.X_text = []
        self.Y_text = []
        for _, row in dataframe.iterrows():
            # retrieving the input and target words
            x = row['2']
            y = row['1']
            if(x == '<\s>' or y == '<\s>'):
                continue
            y = '\t' + y + '\n'
            self.X_text.append(x)
            self.Y_text.append(y)
        # the same train tokenizer is used for validation and testing
        if(train):
            self.ip_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)
            self.ip_tokenizer.fit_on_texts(self.X_text)
            self.op_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)
            self.op_tokenizer.fit_on_texts(self.Y_text)
        
        # tokenizing the text and padding the sequences
        ip_data = self.ip_tokenizer.texts_to_sequences(self.X_text)
        ip_data = tf.keras.preprocessing.sequence.pad_sequences(ip_data,padding='post')
        op_data = self.op_tokenizer.texts_to_sequences(self.Y_text)
        op_data = tf.keras.preprocessing.sequence.pad_sequences(op_data,padding='post')
        if(train):
            self.ip_len = ip_data.shape[1]
            self.op_len = op_data.shape[1]
        # the padding is done on the test and validation data
        if(train == 0):
            ip_data = tf.concat([ip_data,tf.zeros((ip_data.shape[0],self.ip_len-ip_data.shape[1]))],axis=1)
            op_data = tf.concat([op_data,tf.zeros((op_data.shape[0],self.op_len-op_data.shape[1]))],axis=1)
    
        return self.X_text, ip_data, self.ip_tokenizer, self.Y_text, op_data, self.op_tokenizer




