import tensorflow as tf
from attention import Attention

# decoder class for GRU
class GRU_dec(tf.keras.Model):
    def __init__(self, vc_size, embed_dim, dec_dim, bs, dropout=0) -> None:
        super(GRU_dec, self).__init__()
        self.dec_dim = dec_dim
        self.bs = bs
        self.embed = tf.keras.layers.Embedding(vc_size, embed_dim)
        self.gru_model = tf.keras.layers.GRU(dec_dim, return_sequences=True, return_state=True, recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0), dropout=dropout)
        # decoder is followed by a dense layer 
        self.fc_layer = tf.keras.layers.Dense(vc_size)
        # attention values are recieved from the attention class
        self.att = Attention(dec_dim)
    
    def call(self, rand_vec, hidden_st, enc_op):
        # the context vector and attention weights are recieved
        c_vec, att_wt = self.att(hidden_st, enc_op)
        # an embedding of the vector is done
        rand_vec = self.embed(rand_vec)
        # the vector is concatenated with the context vector
        rand_vec = tf.concat([tf.expand_dims(c_vec, 1), rand_vec], axis=-1)
        # the concatenated vector is sent to GRU 
        op, st = self.gru_model(rand_vec)
        op = tf.reshape(op, (-1, op.shape[2]))
        # the output from the gru is then passed to the dense layer
        rand_vec = self.fc_layer(op)

        return rand_vec, st, att_wt