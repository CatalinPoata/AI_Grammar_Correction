import tensorflow

from Decoder import Decoder
from Encoder import Encoder


class EncDecMain(tensorflow.keras.Model):
    def __init__(self, in_voc_sz, out_voc_sz, lstm_sz, in_len, batch_sz, score_fun, att_units, *args):
        super.__init__()

        self.encoder = Encoder(in_voc_sz, lstm_sz, in_len)
        self.decoder = Decoder(out_voc_sz, in_len, lstm_sz, score_fun, att_units)

        self.batch_sz = batch_sz

    def call(self, data):
        input = data[0]
        output = data[1]

        states = self.encoder.init_states(self.batch_sz)
        [enc_out, enc_fs_h, enc_fs_c] = self.encoder(input, states)
        dec_out = self.decoder(output, enc_out, enc_fs_h, enc_fs_c)

        return dec_out