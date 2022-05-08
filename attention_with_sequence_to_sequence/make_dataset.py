import tensorflow as tf
import io

# this class is used to preprocess the data and return the input and ouput tokenizers and data matrices
class PrepareData:
    def __init__(self) -> None:
        self.ip_data = None
        self.op_data = None
        self.ip_tokenizer = None
        self.op_tokenizer = None
    
    # returns all the pairs of input and target words
    def preprocess(self, path):
        raw_data = io.open(path, encoding='UTF-8').read()
        all_data = raw_data.strip().split('\n')
        data = [[self.add_chars(w) for w in single_data.split('\t')[:-1]] for single_data in all_data[:-1]]
        return zip(*data)
    
    # adding a starting and ending token to each word
    def add_chars(self, word):
        word = '\t' + word + '\n'
        return word
    
    # tokenizes the text data
    def make_tokens(self, text_):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)
        tokenizer.fit_on_texts(text_)
        data_matrix = tokenizer.texts_to_sequences(text_)
        data_matrix = tf.keras.preprocessing.sequence.pad_sequences(data_matrix, padding='post')
        return data_matrix, tokenizer
    
    def return_data(self, path):
        op_data, ip_data = self.preprocess(path)
        ip_matrix, ip_tokenizer = self.make_tokens(ip_data)
        op_matrix, op_tokenizer = self.make_tokens(op_data)
        return ip_matrix, op_matrix, ip_tokenizer, op_tokenizer

