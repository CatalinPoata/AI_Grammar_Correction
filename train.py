import datetime
import pickle

import numpy as np
import pandas as pd
import tensorflow
from keras.src.callbacks import ModelCheckpoint
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences

from EncDecMain import EncDecMain


def preprocess_raw(in_file, out_csv):
    m2_file = open(in_file, "r").read().strip().split("\n\n")
    skippable_data = {"noop", "UNK", "Um"}

    correct = []
    incorrect = []

    for data in m2_file:
        corr_sentence = data.split("\n")[0].split(" ")[1:]
        edits = data.split("\n")[1:]
        offset = 0

        for edit in edits:
            edit = edit.split("|||")
            coder = int(edit[-1])

            if (edit[1] not in skippable_data) and (coder == 0):
                edit_data = edit[0].split()[1:]
                start = int(edit_data[0])
                end = int(edit_data[1])
                corr = edit[2].split()
                corr_sentence[start + offset: end + offset] = corr
                offset = offset - (end - start) + len(corr)
        correct.append(" ".join(corr_sentence))

    for data in m2_file:
        temp = data.split("\n")[0].split(" ")[1:]
        temp = " ".join(temp)
        incorrect.append(temp)

    df = pd.DataFrame()

    df["correct"] = correct
    df["incorrect"] = incorrect

    sameSent = []

    for i in range(len(df.values)):
        if df.values[i][0] == df.values[i][1]:
            sameSent.append(i)

    df.drop(sameSent, inplace=True)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    df.to_csv(out_csv, index=False)

    return df


def tokenize_and_pad(df):
    df["correct_inp"] = "$" + df["correct"].astype(str)
    df["correct_out"] = df["correct"].astype(str) + "@"

    tokenizer_inc = Tokenizer(filters="", lower=True, char_level=False)
    tokenizer_inc.fit_on_texts(df["incorrect"].values)

    inc_train = np.array(tokenizer_inc.texts_to_sequences(df["incorrect"].values))

    tokenizer_cor_in = Tokenizer(filters="", char_level=False, lower=False)
    tokenizer_cor_in.fit_on_texts(df["correct_inp"].values)

    cor_train_in = np.array(tokenizer_cor_in.texts_to_sequences(df["correct_inp"].values))

    tokenizer_cor_out = Tokenizer(filters="", char_level=False, lower=False)
    tokenizer_cor_out.fit_on_texts(df["correct_out"].values)

    cor_train_out = np.array(tokenizer_cor_out.texts_to_sequences(df["correct_out"].values))

    with open("token_inc.pickle", "wb") as t1:
        pickle.dump(tokenizer_inc, t1)

    with open("token_cor_in.pickle", "wb") as t2:
        pickle.dump(tokenizer_cor_in, t2)

    with open("token_cor_out.pickle", "wb") as t3:
        pickle.dump(tokenizer_cor_out, t3)

    inc_train = np.array(pad_sequences(inc_train, maxlen=25, padding="post", truncating="post"))
    cor_train_in = np.array(pad_sequences(cor_train_in, maxlen=25, padding="post", truncating="post"))
    cor_train_out = np.array(pad_sequences(cor_train_out, maxlen=25, padding="post", truncating="post"))

    return [df, inc_train, cor_train_in, cor_train_out, len(tokenizer_inc.word_index) + 1, len(tokenizer_cor_out.word_index) + 1]


training_df = preprocess_raw("test.m2", "data_test.csv")
print("Done preprocessing!")

[training_df, inc_train, cor_train_in, cor_train_out, in_voc_sz, out_voc_sz] = tokenize_and_pad(training_df)

in_len = 25
lstm_sz = 64
batch_sz = 256
att_units = 64
score_fun = "dot"

model = EncDecMain(in_voc_sz, out_voc_sz, lstm_sz, in_len, batch_sz, score_fun, att_units)
optimizer = tensorflow.keras.optimizers.Adam()

loss_obj = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tensorflow.math.logical_not(tensorflow.math.equal(real, 0))
    loss = loss_obj(real, pred)
    mask = tensorflow.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tensorflow.reduce_mean(loss)


model.compile(optimizer=optimizer, loss=loss_function())

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

filepath = "checkpoints/weights-{epoch:02d}-{val_loss:.4f}.hdf5"
model_cp = ModelCheckpoint(filepath=filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="auto", save_weights_only=True)

model.fit(x=[inc_train, cor_train_in], y=cor_train_out, epochs=100, batch_size=256, callbacks=[model_cp, tensorboard_callback])





