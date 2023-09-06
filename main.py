import pickle

import numpy as np
import tensorflow
from tensorflow.python.keras.models import load_model

from EncDecMain import EncDecMain


def Predict(input_sentence, model):
    with open("token_cor_out.pickle", "rb") as t1:
        token_cor_out = pickle.load(t1)
    with open("token_inc.pickle", "rb") as t2:
        token_inc = pickle.load(t2)

    cor_dict = token_cor_out.word_index
    inv_cor = {v: k for k, v in cor_dict.items()}

    input_sentence = token_inc.texts_to_sequences([input_sentence])[0]

    init_hs = tensorflow.zeros([1, 64])
    init_cs = tensorflow.zeros([1, 64])

    enc_init_st = [init_hs, init_cs]
    input_sentence = tensorflow.keras.preprocessing.sequence.pad_sequences([input_sentence], maxlen=25, padding="post")[0]

    [enc_out, enc_state_h, enc_state_c] = model.layers[0](np.expand_dims(input_sentence, 0), enc_init_st)

    all = []
    predicted = []
    sent = []

    curr_vec = np.ones((1, 1), dtype="int")

    for i in range(25):
        inf_output, dec_state_h, dec_state_c, att_weights, cont_vect = model.layers[1].osd(curr_vec, enc_out, enc_state_h, enc_state_c)

        enc_state_h = dec_state_h
        enc_state_c = dec_state_c

        curr_vec = np.reshape(np.argmax(inf_output), (1, 1))

        if inv_cor[curr_vec[0][0]] == '@':
            break

        all.append(np.array(inf_output[0]))
        predicted.append(curr_vec[0][0])

    for i in predicted:
        sent.append(inv_cor[i])

    return sent

if __name__ == '__main__':
    model = load_model("Insert model path here")
    while True:
        print("Enter a sentence, please!")
        orig_sent = input()

        corrected_sent = Predict(orig_sent, model)
        print("Predicted correct sentence is:")
        print(corrected_sent)

