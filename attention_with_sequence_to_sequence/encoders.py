import tensorflow as tf

# encoder class for GRU 
class GRU_enc(tf.keras.Model):
    def __init__(self, vc_size, embed_dim, enc_dim, bs, dropout=0) -> None: 
        super(GRU_enc, self).__init__()
        self.enc_dim = enc_dim
        self.bs = bs
        # create the embedding for layer
        self.embed = tf.keras.layers.Embedding(vc_size, embed_dim)
        self.gru_model = tf.keras.layers.GRU(enc_dim, return_sequences=True, return_state=True, recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0), dropout=dropout)
    
    def init_hidden(self):
        # initialize the hidden layer to all zeros 
        return tf.zeros((self.bs, self.enc_dim))

    # this function is called directly by the encoder object
    # initially all zeros hidden state is passed and new values are calculated
    def call(self, ip_batch, hidden_state):
        ip_batch = self.embed(ip_batch)
        op, last_st = self.gru_model(ip_batch, initial_state=hidden_state)
        return op, last_st




    
    
    