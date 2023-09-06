import tensorflow
import Attention
from keras.src.layers import Embedding, RNN, Dense


class OneStepDecoder(tensorflow.keras.Model):
    def __init__(self, voc_sz, in_len, lstm_sz, score_fun, att_units):
        super().__init__()
        self.voc_sz = voc_sz
        self.lstm_sz = lstm_sz
        self.att_units = att_units
        self.score_fun = score_fun
        self.embedding = Embedding(input_dim=voc_sz, output_dim=300, input_length=in_len, mask_zero=True, name="OneStepDecoder_Embedder", trainable=False)
        self.lstm_cell = tensorflow.keras.layers.LSTMCell(lstm_sz)
        self.dec_lstm = RNN(self.lstm_cell, return_sequences=True, return_state=True)
        self.dense = Dense(voc_sz)
        self.attention = Attention(self.score_fun, self.att_units)

    def call(self, dec_in, enc_out, state_h, state_c):
        out1 = self.embedding(dec_in)
        out1 = tensorflow.squeeze(out1, axis=1)
        [cont_vect, att_weights] = self.attention(state_h, enc_out)

        out2 = tensorflow.concat([cont_vect, out1], 1)
        out2 = tensorflow.expand_dims(out2, axis=1)

        dec_out, dec_state_h, dec_state_c = self.dec_lstm(out2, initial_state=[state_h, state_c])

        out3 = self.dense(dec_out)
        out3 = tensorflow.squeeze(out3, axis=1)

        return [out3, dec_state_h, dec_state_c, att_weights, cont_vect]

