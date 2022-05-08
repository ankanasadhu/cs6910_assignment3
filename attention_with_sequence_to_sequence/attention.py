import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, size_) -> None:
        super(Attention, self).__init__()
        self.W = tf.keras.layers.Dense(size_)
        self.U = tf.keras.layers.Dense(size_)
        self.V = tf.keras.layers.Dense(1)

    def call(self, q, v):
        # this stores the scores along the time axis
        q_along_time = tf.expand_dims(q, 1)
        scr = self.V(tf.nn.tanh(self.W(q_along_time) + self.U(v)))
        # attention weights for the values are calculated
        att_wt = tf.nn.softmax(scr, axis=1)
        c_vec = att_wt * v
        c_vec = tf.reduce_sum(c_vec, axis=1)

        return c_vec, att_wt
    

