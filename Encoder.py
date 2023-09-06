import tensorflow
from keras.src.layers import Conv1D, RNN
from tensorflow.python.keras.layers import Embedding


class Encoder(tensorflow.keras.layers.Layer):
    def __init__(self, in_voc_sz, lstm_sz, in_len, **kwargs):
        super().__init__(**kwargs)
        self.lstm_sz = lstm_sz

        self.embed = Embedding(input_dim=in_voc_sz, output_dim=300, input_length = in_len, mask_zero=True, name="Encoder_Embedder", trainable="False")

        self.conv1 = Conv1D(filters=lstm_sz//2, kernel_size=10, activation="relu", kernel_initializer=tensorflow.keras.initializers.HeNormal(), padding="same")
        self.conv2 = Conv1D(filters=lstm_sz, kernel_size=8, activation="relu", kernel_initializer=tensorflow.keras.initializers.HeNormal(), padding="same")

        self.lstm_cell = tensorflow.keras.layers.LSTMCell(lstm_sz)
        self.enc_lstm = RNN(self.lstm_cell, return_sequences=True, return_state=True)

    def call(self, in_seq, states):
        out1 = self.embed(in_seq)
        out2 = self.conv1(out1)
        out2 = self.conv2(out2)
        mask = self.embed.compute_mask(in_seq)
        enc_out, _, enc_state_c = self.enc_lstm(out1, initial_state=states, mask=mask)
        res_out = tensorflow.math.add(enc_out, out2)

        return enc_out, res_out[:, 9, :], enc_state_c

    def init_states(self, batch_size):
        init_hid_state = tensorflow.zeros([batch_size, self.lstm_sz])
        init_cell_state = tensorflow.zeros([batch_size, self.lstm_sz])

        return [init_hid_state, init_cell_state]

