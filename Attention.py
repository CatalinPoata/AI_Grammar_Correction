import tensorflow
from keras.src.layers import Softmax


class Attention(tensorflow.keras.layers.Layer):
    def __init__(self, score_fun, att_units):
        super.__init__()
        self.score_fun = score_fun

        if self.score_fun == "dot":
            self.soft_max = Softmax(axis=1)

    def call(self, dec_hs, enc_out):
        if self.score_fun == "dot":
            att_weight = tensorflow.matmul(enc_out, tensorflow.expand_dims(dec_hs, axis=2))
            cont = tensorflow.matmul(tensorflow.transpose(enc_out, [0, 2, 1]), att_weight)
            cont = tensorflow.squeeze(cont, axis=2)
            output = self.soft_max(att_weight)
            return [cont, output]