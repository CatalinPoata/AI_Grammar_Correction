import tensorflow
import OneStepDecoder

class Decoder(tensorflow.keras.Model):
    def __init__(self, voc_sz, in_len, lstm_sz, score_fun, att_units):
        super().__init__()
        self.voc_sz = voc_sz
        self.lstm_sz = lstm_sz
        self.att_units = att_units
        self.in_len = in_len
        self.score_fun = score_fun
        self.osd = OneStepDecoder(self.voc_sz, self.in_len, self.lstm_sz, self.score_fun, self.att_units)


    @tensorflow.function
    def call(self, dec_in, enc_out, dec_hs, dec_cs):
        outputs = tensorflow.TensorArray(tensorflow.float32, size=dec_in.shape[1])

        for timestep in range(dec_in.shape[1]):
            [out, dec_hs, dec_cs, att_weights, cont_vect] = self.osd(dec_in[:, timestep:timestep+1], enc_out, dec_hs, dec_cs)
            outputs.write(timestep, out)

        outputs = tensorflow.transpose(outputs.stack(), [1, 0, 2])

        return outputs
